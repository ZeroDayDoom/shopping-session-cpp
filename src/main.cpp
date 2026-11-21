// Shopping Sessions: C++17 pipeline. Implements rules.md exactly.
// DATA_DIR in; OUTPUT_DIR/{user_daily,campaign_daily} out (Parquet).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>

#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#ifdef _WIN32
#include <direct.h>
#endif
#ifndef O_BINARY
#define O_BINARY 0
#endif

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

static constexpr i64 INVALID_TS = INT64_MIN;
static constexpr i64 DAY_MS = 86400000ll;
static constexpr i64 LATE_MS = 86400000ll;
static constexpr i64 SEVEN = 7 * DAY_MS;
static constexpr i64 GAP = 1800000ll;
static constexpr u8 PROD_EMPTY = 0x80;
static const size_t NB = 256;

static std::string data_dir = "/data";
static std::string out_dir = "/output";
static unsigned NTHREADS = 8;
static u32 g_unknown_rank = 0;

static std::chrono::steady_clock::time_point T0;
static void logstage(const char *s) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - T0).count();
    fprintf(stderr, "[%8lldms] %s\n", (long long)ms, s);
}

// ---------------------------------------------------------------- hashing

static inline u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// 128-bit key from bytes: two independent multiply streams over 8-byte
// blocks, finalized with mix64. Cheaper than per-block mix64 chains.
static inline void hash128(const u8 *p, size_t n, u64 seed, u64 &h1, u64 &h2) {
    u64 a = seed ^ 0x9E3779B97F4A7C15ull;
    u64 b = mix64(seed + 0x165667B19E3779F9ull);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        u64 v;
        memcpy(&v, p + i, 8);
        a = (a ^ v) * 0xFF51AFD7ED558CCDull;
        b = (b + __builtin_bswap64(v)) * 0xC4CEB9FE1A85EC53ull;
    }
    u64 t = 0;
    if (i < n) {
        memcpy(&t, p + i, n - i);
        t |= (u64)0x80 << ((n - i) * 8); // tail-length tag
    }
    a = (a ^ (t + (u64)n)) * 0xFF51AFD7ED558CCDull;
    b = (b ^ (t * 0x9E3779B97F4A7C15ull)) * 0xC4CEB9FE1A85EC53ull;
    h1 = mix64(a ^ ((u64)n << 1));
    h2 = mix64(b ^ ((u64)n << 1 | 1));
}

static inline u64 hash64(const u8 *p, size_t n) {
    u64 h1, h2;
    hash128(p, n, 0xA1, h1, h2);
    return mix64(h1);
}

// ---------------------------------------------------------------- csv split

// Spec-exact N-field splitter. Fast path: AVX2 scan collects delimiter
// positions, then the spec state machine runs over the positions (verified
// fuzz-equal to the scalar byte-walk). Long lines fall back to a scalar
// delimiter scan + same machine.
template <size_t N>
static inline bool split_fields(const u8 *line, size_t len,
                                size_t (&out)[N][2]) {
    if (len == 0) return false;
    constexpr size_t MAXDEL = (N - 1) + 2 * N + 1;
    u32 hits[MAXDEL + 8];
    size_t nh = 0;
    if (len <= 4096) {
        size_t i = 0;
        while (i < len) {
            size_t nb = std::min<size_t>(32, len - i);
            __m256i v;
            if (nb == 32) {
                v = _mm256_loadu_si256((const __m256i *)(line + i));
            } else {
                u8 tmp[32];
                memcpy(tmp, line + i, nb);
                memset(tmp + nb, 1, 32 - nb); // sentinel: not , or "
                v = _mm256_loadu_si256((const __m256i *)tmp);
            }
            u32 m = (u32)_mm256_movemask_epi8(
                        _mm256_cmpeq_epi8(v, _mm256_set1_epi8(','))) |
                    (u32)_mm256_movemask_epi8(
                        _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"')));
            if (nb < 32) m &= (1u << nb) - 1;
            while (m) {
                int b = __builtin_ctz(m);
                m &= m - 1;
                // a valid N-field line has at most (N-1) commas + 2N quotes
                if (nh >= MAXDEL) return false;
                hits[nh++] = (u32)(i + (size_t)b);
            }
            i += nb;
        }
    } else {
        for (size_t i = 0; i < len; i++) {
            if (line[i] == ',' || line[i] == '"') {
                if (nh >= MAXDEL) return false;
                hits[nh++] = (u32)i;
            }
        }
    }
    size_t fi = 0;
    size_t fs = 0;
    bool quoted = false;
    for (size_t k = 0; k < nh; k++) {
        size_t pos = hits[k];
        if (line[pos] == '"') {
            if (!quoted) {
                if (pos != fs) return false;      // quote inside field
                quoted = true;
                fs = pos + 1;
            } else {
                if (pos + 1 != len && line[pos + 1] != ',') return false;
            }
        } else {
            if (fi >= N) return false;
            if (quoted) {
                if (pos < fs + 1 || line[pos - 1] != '"') return false;
                out[fi][0] = fs;
                out[fi][1] = pos - 1;
            } else {
                out[fi][0] = fs;
                out[fi][1] = pos;
            }
            fi++;
            quoted = false;
            fs = pos + 1;
        }
    }
    if (fi + 1 != N) return false;
    if (quoted) {
        if (len < fs + 1 || line[len - 1] != '"') return false;
        out[fi][0] = fs;
        out[fi][1] = len - 1;
    } else {
        out[fi][0] = fs;
        out[fi][1] = len;
    }
    return true;
}

// ---------------------------------------------------------------- ts parse

static const u32 DIMT[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static inline bool leap(i64 y) {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}
static inline u32 days_in_month(i64 y, i64 m) {
    if (m == 2) return leap(y) ? 29 : 28;
    return DIMT[m - 1];
}
static inline i64 days_from_civil(i64 y, i64 m, i64 d) {
    i64 yy = m <= 2 ? y - 1 : y;
    i64 era = (yy >= 0 ? yy : yy - 399) / 400;
    i64 yoe = yy - era * 400;
    i64 mp = (m + 9) % 12;
    i64 doy = (153 * mp + 2) / 5 + d - 1;
    i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static inline i64 parse_f1(const u8 *b) {
    i64 y = (b[0] - '0') * 1000 + (b[1] - '0') * 100 + (b[2] - '0') * 10 +
            (b[3] - '0');
    i64 mo = (b[5] - '0') * 10 + (b[6] - '0');
    i64 d = (b[8] - '0') * 10 + (b[9] - '0');
    i64 h = (b[11] - '0') * 10 + (b[12] - '0');
    i64 mi = (b[14] - '0') * 10 + (b[15] - '0');
    i64 s = (b[17] - '0') * 10 + (b[18] - '0');
    if (mo < 1 || mo > 12 || d < 1 || d > (i64)days_in_month(y, mo))
        return INVALID_TS;
    if (h > 23 || mi > 59 || s > 59) return INVALID_TS;
    return days_from_civil(y, mo, d) * DAY_MS + h * 3600000 + mi * 60000 +
           s * 1000;
}

static inline i64 parse_ts(const u8 *b, size_t n) {
    if (n == 20 && b[4] == '-' && b[7] == '-' && b[10] == 'T' && b[13] == ':' &&
        b[16] == ':' && b[19] == 'Z') {
        for (size_t di : {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18})
            if (b[di] < '0' || b[di] > '9') return INVALID_TS;
        return parse_f1(b);
    }
    if (n == 25 && (b[19] == '+' || b[19] == '-') && b[22] == ':' &&
        b[4] == '-' && b[7] == '-' && b[10] == 'T' && b[13] == ':' &&
        b[16] == ':') {
        for (size_t di : {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18})
            if (b[di] < '0' || b[di] > '9') return INVALID_TS;
        i64 base = parse_f1(b);
        if (base == INVALID_TS) return INVALID_TS;
        i64 oh = (b[20] - '0') * 10 + (b[21] - '0');
        i64 om = (b[23] - '0') * 10 + (b[24] - '0');
        if (oh > 23 || om > 59) return INVALID_TS;
        i64 off = (oh * 3600 + om * 60) * 1000;
        return b[19] == '+' ? base - off : base + off;
    }
    if (n != 0) {
        bool digits = true;
        for (size_t i = 0; i < n; i++)
            if (b[i] < '0' || b[i] > '9') { digits = false; break; }
        if (digits) {
            i64 v = 0;
            for (size_t i = 0; i < n; i++) {
                int ov = __builtin_mul_overflow(v, 10, &v);
                ov |= __builtin_add_overflow(v, (i64)(b[i] - '0'), &v);
                if (ov) return INT64_MAX;
            }
            return v;
        }
    }
    return INVALID_TS;
}

static inline bool parse_price(const u8 *b, size_t n, i64 &out) {
    if (n == 0) return false;
    i64 v = 0;
    for (size_t i = 0; i < n; i++) {
        if (b[i] < '0' || b[i] > '9') return false;
        v = v * 10 + (b[i] - '0');
        if (v > 10000000) return false; // early exit: also bounds overflow
    }
    if (v < 1) return false;
    out = v;
    return true;
}

static std::string json_get(const std::string &s, const char *key) {
    std::string pat = std::string("\"") + key + "\":";
    size_t i = s.find(pat);
    if (i == std::string::npos) return "";
    size_t q1 = s.find('"', i + pat.size());
    if (q1 == std::string::npos) return "";
    size_t q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return s.substr(q1 + 1, q2 - q1 - 1);
}

// ---------------------------------------------------------------- threads

template <class F>
static void parallel_for(size_t n, size_t grain, const F &f) {
    if (n == 0) return;
    size_t ntasks = (n + grain - 1) / grain;
    if (ntasks > NTHREADS) ntasks = NTHREADS;
    if (ntasks == 1) { f(0, n); return; }
    size_t chunk = (n + ntasks - 1) / ntasks;
    std::vector<std::thread> ts;
    for (size_t t = 0; t < ntasks; t++) {
        size_t a = t * chunk, b = std::min(n, a + chunk);
        if (a >= b) break;
        ts.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto &t : ts) t.join();
}

// ---------------------------------------------------------------- files

struct FileBuf {
    std::vector<u8> data;
    size_t real_len = 0;
};

static std::vector<u8> read_file(const char *path) {
    int fd = open(path, O_RDONLY | O_BINARY);
    std::vector<u8> out;
    if (fd < 0) return out;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return out; }
    size_t len = (size_t)st.st_size;
    out.resize(len + 64, 0);
    size_t got = 0;
    while (got < len) {
        ssize_t r = read(fd, out.data() + got, len - got);
        if (r <= 0) break;
        got += (size_t)r;
    }
    close(fd);
    return out;
}

static void read_all_files(const std::string &dir, std::vector<FileBuf> &out) {
    DIR *d = opendir(dir.c_str());
    std::vector<std::string> paths;
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != nullptr) {
            std::string nm = e->d_name;
            if (nm.size() > 4 && nm.compare(nm.size() - 4, 4, ".csv") == 0) {
                std::string full = dir + "/" + nm;
                struct stat st;
                if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode))
                    paths.push_back(full);
            }
        }
        closedir(d);
    }
    std::sort(paths.begin(), paths.end());
    out.resize(paths.size());
    parallel_for(paths.size(), 1, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; i++) {
            int fd = open(paths[i].c_str(), O_RDONLY | O_BINARY);
            if (fd < 0) continue;
            struct stat st;
            if (fstat(fd, &st) != 0) { close(fd); continue; }
            size_t len = (size_t)st.st_size;
            out[i].data.resize(len + 64, 0); // pad: tail AVX2 loads stay in-bounds
            out[i].real_len = len;
            size_t got = 0;
            while (got < len) {
                ssize_t r = read(fd, out[i].data.data() + got, len - got);
                if (r <= 0) break;
                got += (size_t)r;
            }
            close(fd);
        }
    });
}

// ---------------------------------------------------------------- users

// Lock-free open-addressing table: 64-bit hash -> dense gid (atomic counter).
// A direct-mapped L3-resident cache (full 64-bit tag + gid) absorbs most
// probes; the big table handles insertions and cache misses.
struct UserTable {
    std::atomic<u64> *keys;
    std::atomic<u32> *gids;
    std::atomic<u32> next_id;
    size_t nslots;
    std::atomic<u64> *cache; // packed: (h>>24)<<24 | (gid+1); 0 = empty
    static constexpr size_t CBITS = 21; // 2M entries, 16MB
    explicit UserTable(size_t cap_hint)
        : next_id(0) {
        size_t slots = std::max<size_t>(cap_hint * 2, 1024);
        nslots = 1;
        while (nslots < slots) nslots <<= 1;
        keys = new std::atomic<u64>[nslots];
        gids = new std::atomic<u32>[nslots];
        for (size_t i = 0; i < nslots; i++) {
            keys[i].store(0, std::memory_order_relaxed);
            gids[i].store(UINT32_MAX, std::memory_order_relaxed);
        }
        cache = new std::atomic<u64>[1u << CBITS];
        for (size_t i = 0; i < (1u << CBITS); i++)
            cache[i].store(0, std::memory_order_relaxed);
    }
    ~UserTable() {
        delete[] keys;
        delete[] gids;
        delete[] cache;
    }
    size_t n() const { return (size_t)next_id.load(std::memory_order_relaxed); }
    // Packed cache entry: low 24 bits gid+1, high 40 bits tag. A single
    // atomic store per fill; races only lose a cache line (benign).
    inline void cache_fill(u64 h, u32 gid) {
        size_t ci = (size_t)h & ((1u << CBITS) - 1);
        cache[ci].store((((h >> 24) << 24) | (u64)(gid + 1)),
                        std::memory_order_relaxed);
    }
    inline size_t get_or_insert(u64 h, bool &is_new) {
        size_t ci = (size_t)h & ((1u << CBITS) - 1);
        u64 e = cache[ci].load(std::memory_order_relaxed);
        if (e != 0 && (e >> 24) == (h >> 24)) {
            is_new = false;
            return (size_t)(e & 0xFFFFFFull) - 1;
        }
        size_t i = h & (nslots - 1);
        for (;;) {
            u64 k = keys[i].load(std::memory_order_relaxed);
            if (k == h) {
                // spin until the winner of the CAS has published the gid
                u32 g;
                while ((g = gids[i].load(std::memory_order_relaxed)) ==
                       UINT32_MAX) {
                }
                is_new = false;
                cache_fill(h, g);
                return g;
            }
            if (k == 0) {
                u64 expect = 0;
                if (keys[i].compare_exchange_strong(
                        expect, h, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    u32 id = next_id.fetch_add(1, std::memory_order_relaxed);
                    gids[i].store(id, std::memory_order_relaxed);
                    is_new = true;
                    cache_fill(h, id);
                    return id;
                }
                if (expect == h) {
                    u32 g;
                    while ((g = gids[i].load(std::memory_order_relaxed)) ==
                           UINT32_MAX) {
                    }
                    is_new = false;
                    cache_fill(h, g);
                    return g;
                }
                i = (i + 1) & (nslots - 1);
            } else {
                i = (i + 1) & (nslots - 1);
            }
        }
    }
    inline size_t get(u64 h) const {
        size_t ci = (size_t)h & ((1u << CBITS) - 1);
        u64 e = cache[ci].load(std::memory_order_relaxed);
        if (e != 0 && (e >> 24) == (h >> 24))
            return (size_t)(e & 0xFFFFFFull) - 1;
        size_t i = h & (nslots - 1);
        for (;;) {
            u64 k = keys[i].load(std::memory_order_relaxed);
            if (k == 0) return SIZE_MAX;
            if (k == h) return gids[i].load(std::memory_order_relaxed);
            i = (i + 1) & (nslots - 1);
        }
    }
};

static size_t uf_find(std::vector<u32> &p, size_t x) {
    while (p[x] != (u32)x) {
        p[x] = p[p[x]];
        x = p[x];
    }
    return x;
}

// ---------------------------------------------------------------- records

using BoxStr = std::vector<u8>;
struct Rec3 { u32 gid, off, len; };

struct ERec {
    u64 h1, h2;
    i64 ts, ingest;
    u32 uid, pidx;
    i32 price;
    u8 typ, prod_empty;
    u8 valid;   // 0 = structurally parsed but has invalid field values
};

struct Ev2 {
    u64 key, val;
};

struct CRec {
    u64 h1, h2;
    i64 ts;
    u32 uid, camp, seq, off, len;
};

struct CFin {
    u32 root;
    i64 ts;
    u16 camp;
    u32 arena, off, len;
};

struct Sess {
    u32 root;
    i64 start;
    i32 day;
    i64 revenue;
    bool has_p;
    u16 camp;
};

struct UDRow {
    u32 root;
    i32 day;
    i64 sessions, events, revenue;
    i32 top;
};

// ---------------------------------------------------------------- pair map

// Immutable open-addressing map: 128-bit key -> u32 value. Built once, read
// concurrently.
struct PairMap {
    std::vector<u64> k1, k2;
    std::vector<u32> v;
    size_t mask = 0;
    static constexpr u32 EMPTY = UINT32_MAX;

    void build(const std::vector<u64> &ik1, const std::vector<u64> &ik2,
               const std::vector<u32> &iv) {
        size_t n = ik1.size();
        size_t s = std::max<size_t>(n * 2 + 16, 1024);
        size_t cap = 1;
        while (cap < s) cap <<= 1;
        mask = cap - 1;
        k1.assign(cap, 0);
        k2.assign(cap, 0);
        v.assign(cap, EMPTY);
        for (size_t i = 0; i < n; i++) {
            size_t j = (size_t)mix64(ik1[i]) & mask;
            for (;;) {
                if (v[j] == EMPTY && k1[j] == 0 && k2[j] == 0) {
                    k1[j] = ik1[i];
                    k2[j] = ik2[i];
                    v[j] = iv[i];
                    break;
                }
                if (k1[j] == ik1[i] && k2[j] == ik2[i]) break;
                j = (j + 1) & mask;
            }
        }
    }
    inline u32 get(u64 h1, u64 h2) const {
        size_t j = (size_t)mix64(h1) & mask;
        for (;;) {
            if (k1[j] == 0 && k2[j] == 0) return EMPTY;
            if (k1[j] == h1 && k2[j] == h2) return v[j];
            j = (j + 1) & mask;
        }
    }
};

// ---------------------------------------------------------------- dedup maps

struct EVal {
    i64 ingest, ts;
    u32 uid, seq, pidx;
    i32 price;
    u8 typ, prod_empty;
    u8 valid;
};

// Generation-tagged dedup arena: reusable across buckets without zeroing.
struct EDedup {
    std::vector<u64> k1, k2;
    std::vector<EVal> vals;
    std::vector<u32> gen;
    size_t mask;
    u32 cur = 0;
    void init(size_t rows) {
        size_t s = rows + rows / 2 + 64;
        size_t cap = 64;
        while (cap < s) cap <<= 1;
        mask = cap - 1;
        k1.assign(cap, 0);
        k2.assign(cap, 0);
        vals.resize(cap);
        gen.assign(cap, 0);
        cur = 0;
    }
    inline void upsert(u64 h1, u64 h2, i64 ingest, const EVal &v) {
        size_t i = (size_t)mix64(h1) & mask;
        u32 g = cur;
        for (;;) {
            if (gen[i] != g) {
                gen[i] = g;
                k1[i] = h1;
                k2[i] = h2;
                vals[i] = v;
                vals[i].ingest = ingest;
                return;
            }
            if (k1[i] == h1 && k2[i] == h2) {
                if (ingest < vals[i].ingest) {
                    vals[i] = v;
                    vals[i].ingest = ingest;
                }
                return;
            }
            i = (i + 1) & mask;
        }
    }
    inline void new_bucket() { ++cur; }
    const EVal *slot(u64 h1, u64 h2) const { return nullptr; }
};

struct CVal {
    i64 ts;
    u32 uid, camp, seq, off, len;
};

struct CDedup {
    std::vector<u64> k1, k2;
    std::vector<CVal> vals;
    std::vector<u32> gen;
    size_t mask;
    u32 cur = 0;
    void init(size_t rows) {
        size_t s = rows + rows / 2 + 64;
        size_t cap = 64;
        while (cap < s) cap <<= 1;
        mask = cap - 1;
        k1.assign(cap, 0);
        k2.assign(cap, 0);
        vals.resize(cap);
        gen.assign(cap, 0);
        cur = 0;
    }
    inline void insert_first(u64 h1, u64 h2, const CVal &v) {
        size_t i = (size_t)mix64(h1) & mask;
        u32 g = cur;
        for (;;) {
            if (gen[i] != g) {
                gen[i] = g;
                k1[i] = h1;
                k2[i] = h2;
                vals[i] = v;
                return;
            }
            if (k1[i] == h1 && k2[i] == h2) return;
            i = (i + 1) & mask;
        }
    }
    inline void new_bucket() { ++cur; }
};

// ------------------------------------------------------------- chunk parsers

// Streaming event parser: one AVX2-assisted pass over [lo, hi). Delimiter
// events (',' '"' '\n') drive the spec state machine across the whole chunk;
// a line's field work runs when the line completes. Chunks start and end on
// line boundaries. Buffers are padded (+64 zero bytes) so full 32-byte loads
// are always in-bounds.
static void parse_event_chunk(const u8 *data, size_t lo, size_t hi,
                              UserTable &users, std::vector<u8> &user_arena,
                              std::vector<Rec3> &name_rec,
                              std::vector<u64> &side,
                              std::vector< std::vector<ERec> > &buckets) {
    const size_t est = (hi - lo) / 80 + 64;
    side.reserve(est / 25 + 16);
    if (buckets.size() != NB) buckets.assign(NB, std::vector<ERec>());
    for (auto &b : buckets) {
        b.clear();
        b.reserve(est / NB + 64);
    }
    const size_t bmask = NB - 1;

    size_t fi = 0, fs = lo;
    bool quoted = false, bad = false;
    size_t f[7][2];
    size_t line_start = lo;
    bool first = (lo == 0); // the file's first line is the header

    auto finish_line = [&](size_t endpos) {
        // close the last field
        if (quoted) {
            if (endpos >= fs + 1 && data[endpos - 1] == '"') {
                if (fi < 7) {
                    f[fi][0] = fs;
                    f[fi][1] = endpos - 1;
                }
                fi++;
            } else {
                bad = true;
            }
        } else {
            if (fi < 7) {
                f[fi][0] = fs;
                f[fi][1] = endpos;
            }
            fi++;
        }
        quoted = false;
        if (!first && !bad && fi == 7) {
            size_t len = endpos - line_start;
            const u8 *base = data; // f[] holds absolute offsets
            // Structurally-valid lines ALWAYS enter dedup (keyed by
            // event_id, winner = smallest ingest); only the winner is
            // validated (E5/E20 semantics: a bad duplicate poisons).
            ERec r;
            r.typ = 0;
            r.prod_empty = 0;
            r.pidx = UINT32_MAX;
            r.price = 0;
            r.valid = 1;
            r.uid = UINT32_MAX;

            hash128(base + f[0][0], f[0][1] - f[0][0], 0xEE, r.h1, r.h2);
            if (f[0][1] == f[0][0]) r.valid = 0; // null event_id -> dropped
            if (f[1][1] != f[1][0]) {
                bool is_new;
                r.uid = (u32)users.get_or_insert(
                    hash64(base + f[1][0], f[1][1] - f[1][0]), is_new);
                if (is_new) {
                    Rec3 rc;
                    rc.gid = r.uid;
                    rc.off = (u32)user_arena.size();
                    rc.len = (u32)(f[1][1] - f[1][0]);
                    user_arena.insert(user_arena.end(), base + f[1][0],
                                      base + f[1][1]);
                    name_rec.push_back(rc);
                }
            }
            r.ts = parse_ts(base + f[2][0], f[2][1] - f[2][0]);
            if (r.ts == INVALID_TS) r.valid = 0;
            r.ingest = parse_ts(base + f[3][0], f[3][1] - f[3][0]);
            if (r.ingest == INVALID_TS) r.valid = 0;
            {
                size_t tn = f[4][1] - f[4][0];
                if (tn == 4 && memcmp(base + f[4][0], "view", 4) == 0)
                    r.typ = 1;
                else if (tn == 4 && memcmp(base + f[4][0], "cart", 4) == 0)
                    r.typ = 2;
                else if (tn == 8 &&
                         memcmp(base + f[4][0], "purchase", 8) == 0)
                    r.typ = 3;
                else
                    r.valid = 0;
            }
            if (f[5][1] == f[5][0]) {
                r.prod_empty = 1;
            } else if (r.typ == 3) {
                u64 ph1, ph2;
                hash128(base + f[5][0], f[5][1] - f[5][0], 0xB2, ph1, ph2);
                r.pidx = (u32)(side.size() / 2);
                side.push_back(ph1);
                side.push_back(ph2);
            }
            if (r.typ == 3) {
                i64 v;
                if (!parse_price(base + f[6][0], f[6][1] - f[6][0], v)) {
                    r.valid = 0;
                } else {
                    r.price = (i32)v;
                }
            }
            buckets[(size_t)(r.h1 & (u64)bmask)].push_back(r);
        }
        fi = 0;
        fs = endpos + 1; // next line's first field starts here
        bad = false;
        first = false;
        line_start = endpos + 1;
    };

    size_t i = lo;
    while (i < hi) {
        size_t nb = std::min<size_t>(32, hi - i);
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        u32 mc = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8(',')));
        u32 mq = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"')));
        u32 mn = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n')));
        if (nb < 32) {
            u32 keep = (1u << nb) - 1;
            mc &= keep;
            mq &= keep;
            mn &= keep;
        }
        // No quotes in this block and not inside a quoted field: handle
        // newlines with tight comma loops on either side (the common case).
        if (!mq && !quoted) {
            u32 mrest = mn;
            u32 cmask = mc;
            size_t base = i;
            while (mrest) {
                size_t nn = base + (size_t)__builtin_ctz(mrest);
                mrest &= mrest - 1;
                // commas before this newline
                u32 before = cmask & ((nn - base) ? ((1u << (nn - base)) - 1)
                                                  : 0);
                cmask &= ~((nn - base) ? ((1u << (nn - base)) - 1) : 0);
                while (before) {
                    size_t pos = base + (size_t)__builtin_ctz(before);
                    before &= before - 1;
                    if (bad) continue;
                    if (fi >= 7) {
                        bad = true;
                        continue;
                    }
                    f[fi][0] = fs;
                    f[fi][1] = pos;
                    fi++;
                    fs = pos + 1;
                }
                finish_line(nn);
            }
            // trailing commas after the last newline
            while (cmask) {
                size_t pos = base + (size_t)__builtin_ctz(cmask);
                cmask &= cmask - 1;
                if (bad) continue;
                if (fi >= 7) {
                    bad = true;
                    continue;
                }
                f[fi][0] = fs;
                f[fi][1] = pos;
                fi++;
                fs = pos + 1;
            }
            i += 32;
            continue;
        }
        while (mc | mq | mn) {
            int bc = mc ? __builtin_ctz(mc) : 32;
            int bq = mq ? __builtin_ctz(mq) : 32;
            int bn = mn ? __builtin_ctz(mn) : 32;
            int b = std::min(bc, std::min(bq, bn));
            size_t pos = i + (size_t)b;
            if (b == bn) {
                finish_line(pos);
                mn &= mn - 1;
            } else if (b == bq) {
                if (!quoted) {
                    if (pos != fs) {
                        bad = true;
                    }
                    quoted = true;
                    fs = pos + 1;
                } else {
                    // closing quote: next must be ',' or the line end
                    u8 nc = (pos + 1 < hi) ? data[pos + 1] : 0;
                    if (nc != ',' && nc != '\n' && pos + 1 != hi)
                        bad = true;
                }
                mq &= mq - 1;
            } else {
                if (fi >= 7) {
                    bad = true;
                } else if (!bad) {
                    if (quoted) {
                        if (pos < fs + 1 || data[pos - 1] != '"') {
                            bad = true;
                        } else {
                            f[fi][0] = fs;
                            f[fi][1] = pos - 1;
                        }
                    } else {
                        f[fi][0] = fs;
                        f[fi][1] = pos;
                    }
                    fi++;
                }
                quoted = false;
                fs = pos + 1;
                mc &= mc - 1;
            }
        }
        i += 32;
    }
    if (line_start < hi) finish_line(hi); // final line without trailing \n
}

static void parse_click_chunk(const u8 *data, size_t lo, size_t hi, u32 seq,
                              const PairMap &known_campaigns, UserTable &users,
                              std::vector<u8> &user_arena,
                              std::vector<Rec3> &name_rec,
                              std::vector<u8> &id_arena,
                              std::vector< std::vector<CRec> > &buckets) {
    const size_t est = (hi - lo) / 45 + 64;
    if (buckets.size() != NB) buckets.assign(NB, std::vector<CRec>());
    for (auto &b : buckets) {
        b.clear();
        b.reserve(est / NB + 64);
    }
    const size_t bmask = NB - 1;

    size_t fi = 0, fs = lo;
    bool quoted = false, bad = false;
    size_t f[4][2];
    size_t line_start = lo;
    bool first = (lo == 0);

    auto finish_line = [&](size_t endpos) {
        if (quoted) {
            if (endpos >= fs + 1 && data[endpos - 1] == '"') {
                if (fi < 4) {
                    f[fi][0] = fs;
                    f[fi][1] = endpos - 1;
                }
                fi++;
            } else {
                bad = true;
            }
        } else {
            if (fi < 4) {
                f[fi][0] = fs;
                f[fi][1] = endpos;
            }
            fi++;
        }
        quoted = false;
        if (!first && !bad && fi == 4) {
            const u8 *base = data; // f[] holds absolute offsets
            CRec r;
            r.camp = UINT32_MAX;
            r.seq = seq;
            if (f[0][1] == f[0][0]) {
                // null click_id: structurally fine but droppable; keep out
            } else {
                hash128(base + f[0][0], f[0][1] - f[0][0], 0xDD, r.h1, r.h2);
                r.off = (u32)id_arena.size();
                r.len = (u32)(f[0][1] - f[0][0]);
                id_arena.insert(id_arena.end(), base + f[0][0],
                                base + f[0][1]);
                if (f[1][1] != f[1][0]) {
                    bool is_new;
                    r.uid = (u32)users.get_or_insert(
                        hash64(base + f[1][0], f[1][1] - f[1][0]), is_new);
                    if (is_new) {
                        Rec3 rc;
                        rc.gid = r.uid;
                        rc.off = (u32)user_arena.size();
                        rc.len = (u32)(f[1][1] - f[1][0]);
                        user_arena.insert(user_arena.end(), base + f[1][0],
                                          base + f[1][1]);
                        name_rec.push_back(rc);
                    }
                    u64 ch1, ch2;
                    hash128(base + f[2][0], f[2][1] - f[2][0], 0xC3, ch1, ch2);
                    r.camp = known_campaigns.get(ch1, ch2);
                    r.ts = parse_ts(base + f[3][0], f[3][1] - f[3][0]);
                    if (r.ts != INVALID_TS && r.uid != UINT32_MAX)
                        buckets[(size_t)(r.h1 & (u64)bmask)].push_back(r);
                }
            }
        }
        fi = 0;
        fs = endpos + 1; // next line's first field starts here
        bad = false;
        first = false;
        line_start = endpos + 1;
    };

    size_t i = lo;
    while (i < hi) {
        size_t nb = std::min<size_t>(32, hi - i);
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        u32 mc = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8(',')));
        u32 mq = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"')));
        u32 mn = (u32)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n')));
        if (nb < 32) {
            u32 keep = (1u << nb) - 1;
            mc &= keep;
            mq &= keep;
            mn &= keep;
        }
        while (mc | mq | mn) {
            int bc = mc ? __builtin_ctz(mc) : 32;
            int bq = mq ? __builtin_ctz(mq) : 32;
            int bn = mn ? __builtin_ctz(mn) : 32;
            int b = std::min(bc, std::min(bq, bn));
            size_t pos = i + (size_t)b;
            if (b == bn) {
                finish_line(pos);
                mn &= mn - 1;
            } else if (b == bq) {
                if (!quoted) {
                    if (pos != fs) bad = true;
                    quoted = true;
                    fs = pos + 1;
                } else {
                    u8 nc = (pos + 1 < hi) ? data[pos + 1] : 0;
                    if (nc != ',' && nc != '\n' && pos + 1 != hi) bad = true;
                }
                mq &= mq - 1;
            } else {
                if (fi >= 4) {
                    bad = true;
                } else if (!bad) {
                    if (quoted) {
                        if (pos < fs + 1 || data[pos - 1] != '"') {
                            bad = true;
                        } else {
                            f[fi][0] = fs;
                            f[fi][1] = pos - 1;
                        }
                    } else {
                        f[fi][0] = fs;
                        f[fi][1] = pos;
                    }
                    fi++;
                }
                quoted = false;
                fs = pos + 1;
                mc &= mc - 1;
            }
        }
        i += 32;
    }
    if (line_start < hi) finish_line(hi);
}

static void parse_links(const u8 *data, size_t len, UserTable &users,
                        std::vector<u8> &user_arena,
                        std::vector<Rec3> &name_rec,
                        std::vector<u32> &pairs) {
    // links files are small; keep the scalar byte-walk splitter but avoid
    // re-scanning via memchr+split double passes: single memchr pass is fine
    size_t start = 0;
    bool first = true;
    while (start < len) {
        const u8 *nl = (const u8 *)memchr(data + start, '\n', len - start);
        size_t end = nl ? (size_t)(nl - data) : len;
        if (first) {
            first = false;
            start = end + 1;
            continue;
        }
        size_t llen = end - start;
        if (llen != 0) {
            size_t f[2][2];
            if (split_fields<2>(data + start, llen, f)) {
                if (f[0][1] == f[0][0] || f[1][1] == f[1][0]) {
                    start = end + 1;
                    continue;
                }
                bool na, nb;
                u32 a = (u32)users.get_or_insert(
                    hash64(data + start + f[0][0], f[0][1] - f[0][0]), na);
                u32 b = (u32)users.get_or_insert(
                    hash64(data + start + f[1][0], f[1][1] - f[1][0]), nb);
                if (na) {
                    Rec3 rc;
                    rc.gid = a;
                    rc.off = (u32)user_arena.size();
                    rc.len = (u32)(f[0][1] - f[0][0]);
                    user_arena.insert(user_arena.end(),
                                      data + start + f[0][0],
                                      data + start + f[0][1]);
                    name_rec.push_back(rc);
                }
                if (nb) {
                    Rec3 rc;
                    rc.gid = b;
                    rc.off = (u32)user_arena.size();
                    rc.len = (u32)(f[1][1] - f[1][0]);
                    user_arena.insert(user_arena.end(),
                                      data + start + f[1][0],
                                      data + start + f[1][1]);
                    name_rec.push_back(rc);
                }
                pairs.push_back(a);
                pairs.push_back(b);
            }
        }
        start = end + 1;
    }
}

// ---------------------------------------------------------------- dedup


static void dedup_events_bucket(
    const std::vector< std::pair<u32, const std::vector<ERec> *> > &chunks,
    const std::vector< std::vector<u64> > &sides,
    const std::vector<u32> &root_of_user, u32 uid_empty,
    const PairMap &known_products, u32 unknown_rank, i64 ws, i64 we,
    std::vector<Ev2> &out, EDedup &ed) {
    ed.new_bucket();
    for (auto &cp : chunks) {
        u32 seq = cp.first;
        for (const ERec &r : *cp.second) {
            EVal v;
            v.ingest = r.ingest;
            v.ts = r.ts;
            v.uid = r.uid;
            v.seq = seq;
            v.pidx = r.pidx;
            v.price = r.price;
            v.typ = r.typ;
            v.prod_empty = r.prod_empty;
            v.valid = r.valid;
            ed.upsert(r.h1, r.h2, r.ingest, v);
        }
    }
    for (size_t i = 0; i < ed.gen.size(); i++) {
        if (ed.gen[i] != ed.cur) continue;
        const EVal &v = ed.vals[i];
        if (!v.valid) continue; // dedup winner invalid -> dropped (E5)
        if (v.ts == INVALID_TS || v.ts < ws || v.ts >= we) continue;
        u8 typ = v.typ;
        if (typ == 0 || v.prod_empty) continue;
        if (typ == 3 && v.price <= 0) continue;
        if (v.ingest - v.ts > LATE_MS) continue;
        if (v.uid == UINT32_MAX) continue;
        u32 root = root_of_user[v.uid];
        u32 prank = 0;
        if (typ == 3) {
            u64 ph1 = sides[v.seq][v.pidx * 2];
            u64 ph2 = sides[v.seq][v.pidx * 2 + 1];
            u32 rr = known_products.get(ph1, ph2);
            prank = (rr == PairMap::EMPTY) ? unknown_rank : rr;
        }
        Ev2 e;
        e.key = ((u64)root << 31) | (u64)(v.ts - ws);
        e.val = ((u64)prank << 32) | (u32)v.price;
        out.push_back(e);
    }
}

static void dedup_clicks_bucket(
    const std::vector< std::pair<u32, const std::vector<CRec> *> > &chunks,
    const std::vector<u32> &root_of_user, u32 uid_empty, size_t known_max,
    i64 ws, i64 we, std::vector<CFin> &out, CDedup &cd) {
    cd.new_bucket();
    for (auto &cp : chunks) {
        for (const CRec &r : *cp.second) {
            CVal v;
            v.ts = r.ts;
            v.uid = r.uid;
            v.camp = r.camp;
            v.seq = r.seq;
            v.off = r.off;
            v.len = r.len;
            cd.insert_first(r.h1, r.h2, v);
        }
    }
    for (size_t i = 0; i < cd.gen.size(); i++) {
        if (cd.gen[i] != cd.cur) continue;
        const CVal &v = cd.vals[i];
        if ((i64)v.uid == (i64)uid_empty || v.ts == INVALID_TS ||
            v.ts < ws - SEVEN || v.ts >= we || v.camp >= known_max)
            continue;
        CFin f;
        f.root = root_of_user[v.uid];
        f.ts = v.ts;
        f.camp = (u16)v.camp;
        f.arena = v.seq;
        f.off = v.off;
        f.len = v.len;
        out.push_back(f);
    }
}

// ------------------------------------------------------------ sessionize

static void sessionize_group(const Ev2 *ev, size_t n, i64 ws, u16 organic,
                             std::vector<Sess> &sessions,
                             std::vector<UDRow> &ud_rows) {
    struct SessB {
        u32 root;
        i64 start, last_ts;
        i32 day;
        i64 count, revenue;
        bool has_p;
    };
    struct Run {
        u32 root;
        i32 day;
        i64 sessions, events, revenue;
        std::vector< std::pair<u32, i64> > cats;
        bool has_p;
    };
    SessB sb{};
    bool has_sb = false;
    Run run{};
    bool has_run = false;
    sessions.reserve(n / 4 + 16);
    ud_rows.reserve(n / 8 + 16);

    auto close_session = [&]() {
        run.sessions += 1;
        run.events += sb.count;
        run.revenue += sb.revenue;
        run.has_p |= sb.has_p;
        sessions.push_back(Sess{sb.root, sb.start, sb.day, sb.revenue,
                                sb.has_p, organic});
    };
    auto flush_run = [&]() {
        i32 top = -1;
        if (run.has_p) {
            i64 best_v = INT64_MIN;
            u32 best_k = 0;
            for (auto &kv : run.cats) {
                if (kv.second > best_v ||
                    (kv.second == best_v && kv.first < best_k)) {
                    best_v = kv.second;
                    best_k = kv.first;
                }
            }
            top = (i32)best_k;
        }
        ud_rows.push_back(UDRow{run.root, run.day, run.sessions, run.events,
                                run.revenue, top});
    };
    auto cat_add = [&](u32 rank, i64 amt) {
        for (auto &kv : run.cats) {
            if (kv.first == rank) {
                kv.second += amt;
                return;
            }
        }
        run.cats.emplace_back(rank, amt);
    };

    for (size_t idx = 0; idx < n; idx++) {
        const Ev2 &e = ev[idx];
        u32 root = (u32)(e.key >> 31);
        i64 ts = ws + (i64)(e.key & 0x7FFFFFFFull);
        i32 price = (i32)(e.val & 0xFFFFFFFFull);
        u32 rank = (u32)(e.val >> 32);
        bool new_sess = !has_sb || root != sb.root || ts - sb.last_ts > GAP;
        if (new_sess) {
            if (has_sb) close_session();
            i32 day = (i32)(ts / DAY_MS);
            if (has_run && (run.root != root || run.day != day)) {
                flush_run();
                has_run = false;
            }
            if (!has_run) {
                run.root = root;
                run.day = day;
                run.sessions = 0;
                run.events = 0;
                run.revenue = 0;
                run.cats.clear();
                run.has_p = false;
                has_run = true;
            }
            sb.root = root;
            sb.start = ts;
            sb.last_ts = ts;
            sb.day = day;
            sb.count = 0;
            sb.revenue = 0;
            sb.has_p = false;
            has_sb = true;
        }
        sb.count += 1;
        sb.last_ts = ts;
        if (price != 0) {
            sb.revenue += price;
            sb.has_p = true;
            cat_add(rank, price);
        }
    }
    if (has_sb) close_session();
    if (has_run) flush_run();
}

// ----------------------------------------------------------- attribution

struct Agg {
    std::vector<i64> cred, purc, rev, assi;
    Agg(size_t ncamp1, size_t ndays) {
        cred.assign(ncamp1 * ndays, 0);
        purc.assign(ncamp1 * ndays, 0);
        rev.assign(ncamp1 * ndays, 0);
        assi.assign(ncamp1 * ndays, 0);
    }
    void merge(const Agg &o) {
        for (size_t k = 0; k < cred.size(); k++) {
            cred[k] += o.cred[k];
            purc[k] += o.purc[k];
            rev[k] += o.rev[k];
            assi[k] += o.assi[k];
        }
    }
};

// One root's sessions and that root's clicks; hoisted scratch buffers.
static void attribute_group(Sess *sessions, size_t ns, const CFin *clicks,
                            size_t c_lo, size_t c_hi, u16 organic,
                            size_t ncamp1, size_t ndays, i32 ws_day,
                            u32 tag_base, Agg &agg,
                            std::vector<u32> &counts,
                            std::vector<u8> &in_active,
                            std::vector<u32> &seen_gen,
                            std::vector<u16> &active) {
    size_t l = c_lo;
    i64 r_in = (i64)c_lo - 1;
    for (size_t si = 0; si < ns; si++) {
        Sess &s = sessions[si];
        i64 lo = s.start - SEVEN;
        while (r_in + 1 < (i64)c_hi &&
               clicks[(size_t)(r_in + 1)].ts <= s.start) {
            r_in++;
            u32 c = clicks[(size_t)r_in].camp;
            if (counts[c] == 0 && !in_active[c]) {
                active.push_back((u16)c);
                in_active[c] = 1;
            }
            counts[c]++;
        }
        while ((i64)l <= r_in && clicks[l].ts < lo) {
            u32 c = clicks[l].camp;
            counts[c]--;
            if (counts[c] == 0) in_active[c] = 0;
            l++;
        }
        u16 credited = organic;
        if (r_in >= (i64)l && (size_t)r_in < c_hi &&
            clicks[(size_t)r_in].ts >= lo)
            credited = clicks[(size_t)r_in].camp;
        s.camp = credited;
        if (active.size() > 32)
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [&](u16 c) {
                                            return counts[c] == 0 ||
                                                   !in_active[c];
                                        }),
                         active.end());
        size_t d = (size_t)(s.day - ws_day);
        u32 tag = tag_base + (u32)si;
        for (u16 c16 : active) {
            u32 c = c16;
            if (seen_gen[c] == tag) continue;
            seen_gen[c] = tag;
            if (counts[c] > 0 && in_active[c]) agg.assi[c * ndays + d] += 1;
        }
    }

    for (size_t si = 0; si < ns; si++) {
        const Sess &s = sessions[si];
        size_t d = (size_t)(s.day - ws_day);
        size_t base = (size_t)s.camp * ndays + d;
        agg.cred[base] += 1;
        if (s.has_p) {
            agg.purc[base] += 1;
            agg.rev[base] += s.revenue;
        }
    }
    // reset scratch for the next span
    for (size_t k = c_lo; k < c_hi; k++) {
        u32 c = clicks[k].camp;
        counts[c] = 0;
        in_active[c] = 0;
    }
    active.clear();
}

// Parallel LSD radix sort keyed by a precomputed 64-bit key array
// (16 bits per pass, stable; the key array is permuted alongside).
template <class Rec>
static void parallel_radix_sort(std::vector<u64> &keys, std::vector<Rec> &v) {
    size_t n = keys.size();
    if (n < 2) return;
    u64 maxk = 0;
    for (size_t i = 0; i < n; i++) maxk = std::max(maxk, keys[i]);
    int passes = 1;
    while (passes < 8 && (maxk >> (passes * 16))) passes++;
    size_t nt = std::min<size_t>((size_t)NTHREADS, 32);
    if (nt == 0) nt = 1;
    std::vector<Rec> tmp(n);
    std::vector<u64> tmpkeys(n);
    std::vector<std::vector<u32>> hists(nt);
    std::vector<u32> baset;
    for (int pass = 0; pass < passes; pass++) {
        int shift = pass * 16;
        parallel_for(nt, 1, [&](size_t a, size_t b) {
            for (size_t t = a; t < b; t++) {
                std::vector<u32> &h = hists[t];
                h.assign(65536, 0u);
                size_t lo = n * t / nt, hi = n * (t + 1) / nt;
                for (size_t i = lo; i < hi; i++)
                    h[(keys[i] >> shift) & 0xFFFF]++;
            }
        });
        baset.assign((size_t)nt * 65536, 0u);
        {
            u32 run = 0;
            for (int d = 0; d < 65536; d++) {
                for (size_t t = 0; t < nt; t++) {
                    baset[(size_t)t * 65536 + d] = run;
                    run += hists[t][d];
                }
            }
        }
        parallel_for(nt, 1, [&](size_t a, size_t b) {
            for (size_t t = a; t < b; t++) {
                u32 *mybase = &baset[(size_t)t * 65536];
                size_t lo = n * t / nt, hi = n * (t + 1) / nt;
                for (size_t i = lo; i < hi; i++) {
                    int d = (int)((keys[i] >> shift) & 0xFFFF);
                    size_t dst = mybase[d]++;
                    tmp[dst] = v[i];
                    tmpkeys[dst] = keys[i];
                }
            }
        });
        v.swap(tmp);
        keys.swap(tmpkeys);
    }
}

static void radix_sort_ev2(std::vector<Ev2> &v, std::vector<Ev2> &tmp) {
    size_t n = v.size();
    if (n < 2) return;
    tmp.resize(n);
    std::vector<u32> pos(65536);
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 16;
        std::vector<u32> cnt(65536, 0);
        for (size_t i = 0; i < n; i++)
            cnt[(v[i].key >> shift) & 0xFFFF]++;
        u32 sum = 0;
        for (int b = 0; b < 65536; b++) {
            pos[b] = sum;
            sum += cnt[b];
        }
        for (size_t i = 0; i < n; i++)
            tmp[pos[(v[i].key >> shift) & 0xFFFF]++] = v[i];
        v.swap(tmp);
    }
}

static void radix_sort_cfin(std::vector<CFin> &v, std::vector<CFin> &tmp,
                            i64 click_base) {
    size_t n = v.size();
    if (n < 2) return;
    tmp.resize(n);
    std::vector<u32> pos(65536);
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 16;
        std::vector<u32> cnt(65536, 0);
        for (size_t i = 0; i < n; i++) {
            u64 key = ((u64)v[i].root << 31) |
                      (u64)((v[i].ts - click_base) & 0x7FFFFFFFull);
            cnt[(key >> shift) & 0xFFFF]++;
        }
        u32 sum = 0;
        for (int b = 0; b < 65536; b++) {
            pos[b] = sum;
            sum += cnt[b];
        }
        for (size_t i = 0; i < n; i++) {
            u64 key = ((u64)v[i].root << 31) |
                      (u64)((v[i].ts - click_base) & 0x7FFFFFFFull);
            tmp[pos[(key >> shift) & 0xFFFF]++] = v[i];
        }
        v.swap(tmp);
    }
}

static inline const u8 *id_bytes(const std::vector< std::vector<u8> > &arenas,
                                 const CFin &c) {
    return &arenas[c.arena][c.off];
}

// ---------------------------------------------------------------- parquet

static const char PAR_MAGIC[4] = {'P', 'A', 'R', '1'};

// Minimal Thrift compact protocol writer.
struct TC {
    std::vector<u8> b;
    int last_id = 0;
    void varint(u64 v) {
        while (v >= 0x80) {
            b.push_back((u8)(v | 0x80));
            v >>= 7;
        }
        b.push_back((u8)v);
    }
    void zz32(i32 v) { varint(((u64)v << 1) ^ (u64)(v >> 31)); }
    void zz64(i64 v) { varint(((u64)v << 1) ^ (u64)(v >> 63)); }
    void fhead(int id, int type) {
        int delta = id - last_id;
        if (delta > 0 && delta <= 15)
            b.push_back((u8)((delta << 4) | type));
        else {
            b.push_back((u8)type);
            zz32(id);
        }
        last_id = id;
    }
    void i32f(int id, i32 v) { fhead(id, 5); zz32(v); }
    void i64f(int id, i64 v) { fhead(id, 6); zz64(v); }
    void binf(int id, const std::string &s) {
        fhead(id, 8);
        varint(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void structf(int id) {
        fhead(id, 12);
        id_stack.push_back(last_id); // save enclosing context
        last_id = 0;                 // nested struct starts a fresh sequence
    }
    void rawlist(int etype, size_t n) {
        if (n <= 14)
            b.push_back((u8)((n << 4) | etype));
        else {
            b.push_back((u8)(0xF0 | etype));
            varint(n);
        }
    }
    void listf(int id, int etype, size_t n) {
        fhead(id, 9);
        rawlist(etype, n);
    }
    void stop() {
        b.push_back(0);
        last_id = id_stack.empty() ? 0 : id_stack.back();
        if (!id_stack.empty()) id_stack.pop_back();
    }
    // Structs nested inside thrift lists restart their own field-id sequence.
    void elem_start() {
        id_stack.push_back(last_id);
        last_id = 0;
    }
    std::vector<int> id_stack;
};

// thrift compact type codes
static const int CT_I32 = 5, CT_I64 = 6, CT_BINARY = 8, CT_STRUCT = 12;
// parquet enums
static const int PT_INT32 = 1, PT_INT64 = 2, PT_BYTE_ARRAY = 6;
static const int CV_UTF8 = 0, CV_DATE = 6;
static const int REP_REQUIRED = 0, REP_OPTIONAL = 1;
static const int ENC_PLAIN = 0, ENC_RLE = 3;
static const int CODEC_UNCOMPRESSED = 0;

struct SchemaField {
    const char *name;
    int ptype;
    int rep;
    int converted;
};

struct ColChunkInfo {
    int ptype;
    const char *name;
    int converted;
    i64 data_page_offset;
    i64 total_size;
    i64 num_values;
};

static void fwrite_all(FILE *f, const void *p, size_t n) {
    if (n) fwrite(p, 1, n, f);
}

static void encode_def_levels(std::vector<u8> &out, const std::vector<u8> &lv) {
    size_t n = lv.size();
    bool all1 = true, all0 = true;
    for (u8 v : lv) {
        if (v) all0 = false;
        else all1 = false;
    }
    TC t;
    if (n > 0 && all1) {
        t.varint((u64)n << 1);
        t.b.push_back(1);
    } else if (n > 0 && all0) {
        t.varint((u64)n << 1);
        t.b.push_back(0);
    } else {
        size_t groups = (n + 7) / 8;
        t.varint(((u64)groups << 1) | 1);
        for (size_t g = 0; g < groups; g++) {
            u8 byte = 0;
            for (size_t k = 0; k < 8 && g * 8 + k < n; k++)
                if (lv[g * 8 + k]) byte |= (u8)(1u << k);
            t.b.push_back(byte);
        }
    }
    // v1 data pages: the levels section is prefixed with its 4-byte length.
    u32 lvlen = (u32)t.b.size();
    out.insert(out.end(), (u8 *)&lvlen, (u8 *)&lvlen + 4);
    out.insert(out.end(), t.b.begin(), t.b.end());
}

static std::vector<u8> page_header(i64 num_values, size_t body_len) {
    TC ph;
    ph.i32f(1, 0);  // DATA_PAGE
    ph.i32f(2, (i32)body_len);
    ph.i32f(3, (i32)body_len);
    ph.structf(5);
    ph.i32f(1, (i32)num_values);
    ph.i32f(2, ENC_PLAIN);
    ph.i32f(3, ENC_RLE);
    ph.i32f(4, ENC_RLE);
    ph.stop();
    ph.stop();
    return ph.b;
}

static ColChunkInfo emit_column(FILE *f, int ptype, const char *name,
                                int converted, i64 num_values,
                                const std::vector<u8> &body) {
    std::vector<u8> hdr = page_header(num_values, body.size());
    i64 off = ftell(f);
    fwrite_all(f, hdr.data(), hdr.size());
    fwrite_all(f, body.data(), body.size());
    ColChunkInfo ci;
    ci.ptype = ptype;
    ci.name = name;
    ci.converted = converted;
    ci.data_page_offset = off;
    ci.total_size = (i64)(hdr.size() + body.size());
    ci.num_values = num_values;
    return ci;
}

static std::vector<u8> page_i32(const std::vector<i32> &v) {
    std::vector<u8> b(v.size() * 4);
    memcpy(b.data(), v.data(), v.size() * 4);
    return b;
}
static std::vector<u8> page_i64(const std::vector<i64> &v) {
    std::vector<u8> b(v.size() * 8);
    memcpy(b.data(), v.data(), v.size() * 8);
    return b;
}
static std::vector<u8> page_bytes_req(
    const std::vector< std::vector<u8> > &v) {
    std::vector<u8> b;
    size_t tot = 0;
    for (auto &s : v) tot += 4 + s.size();
    b.resize(tot);
    size_t w = 0;
    for (auto &s : v) {
        u32 n = (u32)s.size();
        memcpy(&b[w], &n, 4);
        w += 4;
        if (n) memcpy(&b[w], s.data(), n);
        w += n;
    }
    return b;
}
static std::vector<u8> page_bytes_opt(
    const std::vector<u8> &levels, const std::vector< std::vector<u8> > &v) {
    std::vector<u8> b;
    size_t tot = 0;
    for (auto &s : v) tot += 4 + s.size();
    std::vector<u8> lv;
    encode_def_levels(lv, levels);
    b.resize(lv.size() + tot);
    memcpy(b.data(), lv.data(), lv.size());
    size_t w = lv.size();
    for (auto &s : v) {
        u32 n = (u32)s.size();
        memcpy(&b[w], &n, 4);
        w += 4;
        if (n) memcpy(&b[w], s.data(), n);
        w += n;
    }
    return b;
}

static std::vector<u8> build_footer(
    i64 num_rows, const std::vector<SchemaField> &fields,
    const std::vector<ColChunkInfo> &chunks, i64 rg_total_size) {
    TC tc;
    tc.i32f(1, 1);
    tc.listf(2, CT_STRUCT, fields.size() + 1);
    tc.elem_start();
    tc.binf(4, "schema");
    tc.i32f(5, (i32)fields.size());
    tc.stop();
    for (auto &f : fields) {
        tc.elem_start();
        tc.i32f(1, f.ptype);
        tc.i32f(3, f.rep);
        tc.binf(4, f.name);
        if (f.converted >= 0) tc.i32f(6, f.converted);
        tc.stop();
    }
    tc.i64f(3, num_rows);
    tc.listf(4, CT_STRUCT, 1);
    tc.elem_start();
    tc.listf(1, CT_STRUCT, chunks.size());
    for (auto &c : chunks) {
        tc.elem_start();
        tc.i64f(2, c.data_page_offset);
        tc.structf(3);
        tc.i32f(1, c.ptype);
        tc.listf(2, CT_I32, 2);
        tc.zz32(ENC_PLAIN);
        tc.zz32(ENC_RLE);
        tc.listf(3, CT_BINARY, 1);
        tc.varint(strlen(c.name));
        tc.b.insert(tc.b.end(), c.name, c.name + strlen(c.name));
        tc.i32f(4, CODEC_UNCOMPRESSED);
        tc.i64f(5, c.num_values);
        tc.i64f(6, c.total_size);
        tc.i64f(7, c.total_size);
        tc.i64f(9, c.data_page_offset);
        tc.stop();
        tc.stop();
    }
    tc.i64f(2, rg_total_size);
    tc.i64f(3, num_rows);
    tc.stop();
    tc.binf(6, "shopping-session-cpp");
    tc.stop();
    return tc.b;
}

using BoxStr = std::vector<u8>;

struct SchemaFieldX {
    const char *name;
    int ptype;
    int rep;
    int converted;
};

static void write_user_daily_parquet(
    const std::string &path, const std::vector<UDRow> &rows,
    const std::vector<BoxStr> &name_by_gid,
    const std::vector<u32> &name_of_root,
    const std::vector<BoxStr> &categories) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); exit(1); }
    fwrite_all(f, PAR_MAGIC, 4);

    std::vector<ColChunkInfo> infos;
    i64 rg_size = 0;
    i64 nrows = (i64)rows.size();

    {
        std::vector<u8> body;
        size_t tot = 0;
        for (size_t i = 0; i < rows.size(); i++)
            tot += 4 + name_by_gid[name_of_root[rows[i].root]].size();
        body.resize(tot);
        size_t w = 0;
        for (size_t i = 0; i < rows.size(); i++) {
            const BoxStr &s = name_by_gid[name_of_root[rows[i].root]];
            u32 n = (u32)s.size();
            memcpy(&body[w], &n, 4);
            w += 4;
            if (n) memcpy(&body[w], s.data(), n);
            w += n;
        }
        ColChunkInfo ci = emit_column(f, PT_BYTE_ARRAY, "user_id", CV_UTF8,
                                      nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i32> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].day;
        std::vector<u8> body = page_i32(v);
        ColChunkInfo ci = emit_column(f, PT_INT32, "day", CV_DATE, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].sessions;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "sessions", -1, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].events;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "events", -1, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].revenue;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "revenue_cents", -1, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<u8> levels(rows.size());
        size_t tot = 0;
        for (size_t i = 0; i < rows.size(); i++) {
            if (rows[i].top >= 0) {
                levels[i] = 1;
                tot += 4 + categories[(size_t)rows[i].top].size();
            } else {
                levels[i] = 0;
            }
        }
        std::vector<u8> lv;
        encode_def_levels(lv, levels);
        std::vector<u8> body;
        body.resize(lv.size() + tot);
        memcpy(body.data(), lv.data(), lv.size());
        size_t w = lv.size();
        for (size_t i = 0; i < rows.size(); i++) {
            if (rows[i].top >= 0) {
                const BoxStr &s = categories[(size_t)rows[i].top];
                u32 n = (u32)s.size();
                memcpy(&body[w], &n, 4);
                w += 4;
                if (n) memcpy(&body[w], s.data(), n);
                w += n;
            }
        }
        ColChunkInfo ci = emit_column(f, PT_BYTE_ARRAY, "top_category",
                                      CV_UTF8, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }

    std::vector<SchemaField> fields = {
        {"user_id", PT_BYTE_ARRAY, REP_REQUIRED, CV_UTF8},
        {"day", PT_INT32, REP_REQUIRED, CV_DATE},
        {"sessions", PT_INT64, REP_REQUIRED, -1},
        {"events", PT_INT64, REP_REQUIRED, -1},
        {"revenue_cents", PT_INT64, REP_REQUIRED, -1},
        {"top_category", PT_BYTE_ARRAY, REP_OPTIONAL, CV_UTF8},
    };
    std::vector<u8> footer = build_footer(nrows, fields, infos, rg_size);
    u32 flen = (u32)footer.size();
    fwrite_all(f, footer.data(), footer.size());
    fwrite_all(f, &flen, 4);
    fwrite_all(f, PAR_MAGIC, 4);
    fclose(f);
}

static void write_campaign_daily_parquet(
    const std::string &path, const std::vector<BoxStr> &camp_ids,
    const std::vector<BoxStr> &camp_channels, const std::vector<i64> &cred,
    const std::vector<i64> &purc, const std::vector<i64> &rev,
    const std::vector<i64> &assi, size_t ncamp1, size_t ndays, i32 ws_day) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); exit(1); }
    fwrite_all(f, PAR_MAGIC, 4);

    struct RowData {
        const BoxStr *id, *ch;
        i32 day;
        i64 sessions, purchasing, revenue, assisted;
    };
    std::vector<RowData> rows;
    for (size_t c = 0; c < ncamp1; c++) {
        for (size_t d = 0; d < ndays; d++) {
            size_t idx = c * ndays + d;
            if (cred[idx] > 0 || assi[idx] > 0) {
                RowData r;
                r.id = &camp_ids[c];
                r.ch = &camp_channels[c];
                r.day = ws_day + (i32)d;
                r.sessions = cred[idx];
                r.purchasing = purc[idx];
                r.revenue = rev[idx];
                r.assisted = assi[idx];
                rows.push_back(r);
            }
        }
    }
    i64 nrows = (i64)rows.size();

    std::vector<ColChunkInfo> infos;
    i64 rg_size = 0;
    {
        std::vector< std::vector<u8> > vals(rows.size());
        for (size_t i = 0; i < rows.size(); i++) vals[i] = *rows[i].id;
        std::vector<u8> body = page_bytes_req(vals);
        ColChunkInfo ci = emit_column(f, PT_BYTE_ARRAY, "campaign_id", CV_UTF8,
                                      nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector< std::vector<u8> > vals(rows.size());
        for (size_t i = 0; i < rows.size(); i++) vals[i] = *rows[i].ch;
        std::vector<u8> body = page_bytes_req(vals);
        ColChunkInfo ci = emit_column(f, PT_BYTE_ARRAY, "channel", CV_UTF8,
                                      nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i32> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].day;
        std::vector<u8> body = page_i32(v);
        ColChunkInfo ci = emit_column(f, PT_INT32, "day", CV_DATE, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].sessions;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "sessions", -1, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].purchasing;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "purchasing_sessions", -1,
                                      nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].revenue;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "revenue_cents", -1, nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }
    {
        std::vector<i64> v(rows.size());
        for (size_t i = 0; i < rows.size(); i++) v[i] = rows[i].assisted;
        std::vector<u8> body = page_i64(v);
        ColChunkInfo ci = emit_column(f, PT_INT64, "assisted_sessions", -1,
                                      nrows, body);
        rg_size += ci.total_size;
        infos.push_back(ci);
    }

    std::vector<SchemaField> fields = {
        {"campaign_id", PT_BYTE_ARRAY, REP_REQUIRED, CV_UTF8},
        {"channel", PT_BYTE_ARRAY, REP_REQUIRED, CV_UTF8},
        {"day", PT_INT32, REP_REQUIRED, CV_DATE},
        {"sessions", PT_INT64, REP_REQUIRED, -1},
        {"purchasing_sessions", PT_INT64, REP_REQUIRED, -1},
        {"revenue_cents", PT_INT64, REP_REQUIRED, -1},
        {"assisted_sessions", PT_INT64, REP_REQUIRED, -1},
    };
    std::vector<u8> footer = build_footer(nrows, fields, infos, rg_size);
    u32 flen = (u32)footer.size();
    fwrite_all(f, footer.data(), footer.size());
    fwrite_all(f, &flen, 4);
    fwrite_all(f, PAR_MAGIC, 4);
    fclose(f);
}



// ---------------------------------------------------------------- main

static void mkdirs(const std::string &d) {
#ifdef _WIN32
    mkdir(d.c_str());
#else
    mkdir(d.c_str(), 0777);
#endif
}

int main() {
    T0 = std::chrono::steady_clock::now();
    {
        const char *e = getenv("DATA_DIR");
        if (e) data_dir = e;
        e = getenv("OUTPUT_DIR");
        if (e) out_dir = e;
        e = getenv("NCPU");
        NTHREADS = e ? (unsigned)atoi(e)
                     : (unsigned)std::thread::hardware_concurrency();
        if (NTHREADS == 0) NTHREADS = 8;
        if (NTHREADS > 32) NTHREADS = 32;
    }
    logstage("start");

    // ---- manifest
    std::vector<u8> man = read_file((data_dir + "/manifest.json").c_str());
    std::string ms((const char *)man.data(), man.size());
    std::string ws_s = json_get(ms, "window_start");
    std::string we_s = json_get(ms, "window_end");
    i64 ws = INVALID_TS, we = INVALID_TS;
    {
        std::vector<u8> a(ws_s.begin(), ws_s.end());
        std::vector<u8> b(we_s.begin(), we_s.end());
        ws = parse_ts(a.data(), a.size());
        we = parse_ts(b.data(), b.size());
    }
    if (ws == INVALID_TS || we == INVALID_TS || we <= ws) {
        fprintf(stderr, "bad manifest\n");
        return 1;
    }
    i32 ws_day = (i32)(ws / DAY_MS);
    i32 we_day = (i32)(we / DAY_MS);
    size_t ndays = (size_t)(we_day - ws_day + 1);
    logstage("manifest");

    // ---- campaigns (single csv at DATA_DIR root)
    std::vector<BoxStr> camp_ids;
    std::vector<BoxStr> camp_channels;
    std::unordered_map<std::string, u32> camp_idx;
    {
        std::vector<u8> t = read_file((data_dir + "/campaigns.csv").c_str());
        size_t pos = 0;
        bool first = true;
        while (pos < t.size()) {
            const u8 *nl = (const u8 *)memchr(t.data() + pos, '\n', t.size() - pos);
            size_t end = nl ? (size_t)(nl - t.data()) : t.size();
            size_t llen = end - pos;
            if (llen && t[end - 1] == '\r') llen--;
            if (first) {
                first = false;
            } else if (llen != 0) {
                size_t f[2][2];
                if (split_fields<2>(t.data() + pos, llen, f)) {
                    std::string id((const char *)t.data() + pos + f[0][0],
                                   f[0][1] - f[0][0]);
                    auto it = camp_idx.find(id);
                    u32 idx;
                    if (it == camp_idx.end()) {
                        idx = (u32)camp_ids.size();
                        camp_idx.emplace(id, idx);
                        camp_ids.push_back(BoxStr(t.data() + pos + f[0][0],
                                                  t.data() + pos + f[0][1]));
                    } else {
                        idx = it->second;
                    }
                    if (camp_channels.size() <= idx)
                        camp_channels.resize(idx + 1);
                    camp_channels[idx].assign(t.data() + pos + f[1][0],
                                              t.data() + pos + f[1][1]);
                }
            }
            pos = end + 1;
        }
    }
    while (camp_channels.size() < camp_ids.size())
        camp_channels.push_back(BoxStr());
    size_t n_camp = camp_ids.size();
    u16 organic = (u16)n_camp;
    size_t ncamp1 = n_camp + 1;
    PairMap known_campaigns;
    {
        std::vector<u64> ck1, ck2;
        std::vector<u32> cv;
        ck1.reserve(n_camp); ck2.reserve(n_camp); cv.reserve(n_camp);
        for (u32 c = 0; c < (u32)n_camp; c++) {
            u64 h1, h2;
            hash128(camp_ids[c].data(), camp_ids[c].size(), 0xC3, h1, h2);
            ck1.push_back(h1);
            ck2.push_back(h2);
            cv.push_back(c);
        }
        known_campaigns.build(ck1, ck2, cv);
    }
    logstage("campaigns");

    // ---- products: product hash -> category rank
    std::vector<BoxStr> categories;
    PairMap known_products;
    {
        std::vector<u8> t = read_file((data_dir + "/products.csv").c_str());
        std::unordered_map<std::string, u32> prod_cat; // id -> cat idx
        std::vector<BoxStr> prod_ids;
        std::vector<u32> prod_cat_idx;
        std::vector<BoxStr> cat_names;
        std::unordered_map<std::string, u32> cat_idx;
        size_t pos = 0;
        bool first = true;
        while (pos < t.size()) {
            const u8 *nl = (const u8 *)memchr(t.data() + pos, '\n', t.size() - pos);
            size_t end = nl ? (size_t)(nl - t.data()) : t.size();
            size_t llen = end - pos;
            if (llen && t[end - 1] == '\r') llen--;
            if (first) {
                first = false;
            } else if (llen != 0) {
                size_t f[2][2];
                if (split_fields<2>(t.data() + pos, llen, f)) {
                    std::string pid((const char *)t.data() + pos + f[0][0],
                                    f[0][1] - f[0][0]);
                    std::string cat((const char *)t.data() + pos + f[1][0],
                                    f[1][1] - f[1][0]);
                    auto ci = cat_idx.find(cat);
                    u32 cidx;
                    if (ci == cat_idx.end()) {
                        cidx = (u32)cat_names.size();
                        cat_idx.emplace(cat, cidx);
                        cat_names.push_back(BoxStr(cat.begin(), cat.end()));
                    } else {
                        cidx = ci->second;
                    }
                    if (prod_cat.find(pid) == prod_cat.end()) {
                        prod_cat.emplace(pid, cidx);
                        prod_ids.push_back(BoxStr(t.data() + pos + f[0][0],
                                                  t.data() + pos + f[0][1]));
                        prod_cat_idx.push_back(cidx);
                    }
                }
            }
            pos = end + 1;
        }
        categories = cat_names;
        std::sort(categories.begin(), categories.end(),
                  [](const BoxStr &x, const BoxStr &y) {
                      size_t mn = std::min(x.size(), y.size());
                      int c = memcmp(x.data(), y.data(), mn);
                      return c != 0 ? c < 0 : x.size() < y.size();
                  });
        // unknown products: 'unknown' competes byte-wise like any category
        // (rules: a tie in top_category goes to the byte-wise smallest)
        categories.push_back(BoxStr{'u', 'n', 'k', 'n', 'o', 'w', 'n'});
        std::sort(categories.begin(), categories.end(),
                  [](const BoxStr &x, const BoxStr &y) {
                      size_t mn = std::min(x.size(), y.size());
                      int c = memcmp(x.data(), y.data(), mn);
                      return c != 0 ? c < 0 : x.size() < y.size();
                  });
        std::unordered_map<std::string, u32> sorted_rank;
        for (size_t k = 0; k < categories.size(); k++)
            sorted_rank[std::string(categories[k].begin(),
                                    categories[k].end())] = (u32)k;
        std::vector<u64> pk1, pk2;
        std::vector<u32> pv;
        pk1.reserve(prod_ids.size());
        pk2.reserve(prod_ids.size());
        pv.reserve(prod_ids.size());
        for (size_t i = 0; i < prod_ids.size(); i++) {
            u64 h1, h2;
            hash128(prod_ids[i].data(), prod_ids[i].size(), 0xB2, h1, h2);
            pk1.push_back(h1);
            pk2.push_back(h2);
            std::string cn(cat_names[prod_cat_idx[i]].begin(),
                           cat_names[prod_cat_idx[i]].end());
            pv.push_back(sorted_rank[cn]);
        }
        known_products.build(pk1, pk2, pv);
        u32 unk = sorted_rank["unknown"]; // 'unknown' sorted byte-wise in
        g_unknown_rank = unk;
    }
    u32 unknown_rank = g_unknown_rank;
    logstage("products");

    // ---- users
    UserTable users(3000000);
    std::vector<u8> user_arena;
    std::vector<Rec3> all_name_rec;
    std::vector<u32> link_pairs;
    {
        std::vector<FileBuf> fb;
        read_all_files(data_dir + "/links", fb);
        size_t nlf = fb.size();
        std::vector< std::vector<u8> > la(nlf);
        std::vector< std::vector<Rec3> > lr(nlf);
        std::vector< std::vector<u32> > lp(nlf);
        parallel_for(nlf, 1, [&](size_t a, size_t b) {
            for (size_t i = a; i < b; i++) {
                if (fb[i].data.empty()) continue;
                parse_links(fb[i].data.data(), fb[i].real_len, users,
                            la[i], lr[i], lp[i]);
            }
        });
        for (size_t i = 0; i < nlf; i++) {
            size_t base = user_arena.size();
            user_arena.insert(user_arena.end(), la[i].begin(), la[i].end());
            for (auto &rc : lr[i]) {
                rc.off += (u32)base;
                all_name_rec.push_back(rc);
            }
        }
        size_t np = 0;
        for (auto &v : lp) np += v.size();
        link_pairs.reserve(np);
        for (auto &v : lp) link_pairs.insert(link_pairs.end(), v.begin(), v.end());
    }
    logstage("links parsed");

    // ---- events
    std::vector<FileBuf> ebufs;
    read_all_files(data_dir + "/events", ebufs);
    // One task per 32MB line-aligned chunk: each chunk gets its OWN bucket
    // vectors (a file's later chunks must not wipe earlier ones).
    struct ETask { u32 file; size_t off, end; };
    std::vector<ETask> etasks;
    for (size_t i = 0; i < ebufs.size(); i++) {
        size_t len = ebufs[i].real_len, off = 0;
        while (off < len) {
            size_t end = std::min(len, off + (size_t)(32 << 20));
            if (end < len) {
                const u8 *nl = (const u8 *)memchr(
                    ebufs[i].data.data() + end, '\n', len - end);
                if (nl) end = (size_t)(nl - ebufs[i].data.data()) + 1;
                else end = len;
            }
            etasks.push_back(ETask{(u32)i, off, end});
            off = end;
        }
    }
    size_t n_echunks = etasks.size();
    std::vector< std::vector<u64> > esides(n_echunks);
    std::vector< std::vector< std::vector<ERec> > > ebuckets(n_echunks);
    std::vector< std::vector<u8> > ev_arenas(n_echunks);
    std::vector< std::vector<Rec3> > ev_name_recs(n_echunks);
    parallel_for(n_echunks, 1, [&](size_t a, size_t b) {
        for (size_t t = a; t < b; t++) {
            const ETask &tk = etasks[t];
            parse_event_chunk(ebufs[tk.file].data.data(), tk.off, tk.end,
                              users, ev_arenas[t], ev_name_recs[t],
                              esides[t], ebuckets[t]);
        }
    });
    logstage("events parsed");

    // ---- clicks
    std::vector<FileBuf> cbufs;
    read_all_files(data_dir + "/clicks", cbufs);
    struct CTask { u32 file; size_t off, end; };
    std::vector<CTask> ctasks;
    for (size_t i = 0; i < cbufs.size(); i++) {
        size_t len = cbufs[i].real_len, off = 0;
        while (off < len) {
            size_t end = std::min(len, off + (size_t)(32 << 20));
            if (end < len) {
                const u8 *nl = (const u8 *)memchr(
                    cbufs[i].data.data() + end, '\n', len - end);
                if (nl) end = (size_t)(nl - cbufs[i].data.data()) + 1;
                else end = len;
            }
            ctasks.push_back(CTask{(u32)i, off, end});
            off = end;
        }
    }
    size_t n_cchunks = ctasks.size();
    std::vector< std::vector< std::vector<CRec> > > cbuckets(n_cchunks);
    std::vector< std::vector<u8> > cl_arenas(n_cchunks);
    std::vector< std::vector<Rec3> > cl_name_recs(n_cchunks);
    parallel_for(n_cchunks, 1, [&](size_t a, size_t b) {
        for (size_t t = a; t < b; t++) {
            const CTask &tk = ctasks[t];
            parse_click_chunk(cbufs[tk.file].data.data(), tk.off, tk.end,
                              (u32)t, known_campaigns, users, cl_arenas[t],
                              cl_name_recs[t], cl_arenas[t], cbuckets[t]);
        }
    });
    logstage("clicks parsed");

    // ---- reconcile names (per-chunk arenas; links first for priority)
    size_t n_ids = users.n();
    std::vector<BoxStr> name_by_gid(n_ids);
    for (auto &rc : all_name_rec) {
        if (name_by_gid[rc.gid].empty())
            name_by_gid[rc.gid].assign(user_arena.begin() + rc.off,
                                       user_arena.begin() + rc.off + rc.len);
    }
    // event-chunk and click-chunk name records are disjoint per gid (is_new
    // guarantees a single recorder), so the two loops parallelize safely
    {
        std::vector<size_t> eq(n_echunks), cq(n_cchunks);
        for (size_t ci = 0; ci < n_echunks; ci++) eq[ci] = ci;
        for (size_t ci = 0; ci < n_cchunks; ci++) cq[ci] = ci;
        auto fill = [&](std::vector<BoxStr> &arenas,
                        std::vector< std::vector<Rec3> > &recs,
                        std::vector<size_t> &idx) {
            parallel_for(idx.size(), 1, [&](size_t a, size_t b) {
                for (size_t t = a; t < b; t++) {
                    size_t ci = idx[t];
                    for (auto &rc : recs[ci]) {
                        if (name_by_gid[rc.gid].empty())
                            name_by_gid[rc.gid].assign(
                                arenas[ci].begin() + rc.off,
                                arenas[ci].begin() + rc.off + rc.len);
                    }
                }
            });
        };
        fill(ev_arenas, ev_name_recs, eq);
        fill(cl_arenas, cl_name_recs, cq);
    }
    logstage("names reconciled");

    // ---- union-find links
    std::vector<u32> uf(n_ids);
    for (size_t i = 0; i < n_ids; i++) uf[i] = (u32)i;
    for (size_t k = 0; k + 1 < link_pairs.size(); k += 2) {
        size_t ra = uf_find(uf, link_pairs[k]);
        size_t rb = uf_find(uf, link_pairs[k + 1]);
        if (ra != rb) uf[std::max(ra, rb)] = (u32)std::min(ra, rb);
    }
    std::vector<u32> root_of_user(n_ids);
    for (size_t i = 0; i < n_ids; i++) root_of_user[i] = uf_find(uf, i);
    bool has_empty;
    size_t uid_empty = users.get(hash64((const u8 *)"", 0));
    has_empty = (uid_empty != SIZE_MAX);
    u32 uid_empty_val = has_empty ? (u32)uid_empty : UINT32_MAX;
    std::vector<u32> name_of_root(n_ids, UINT32_MAX);
    for (size_t i = 0; i < n_ids; i++) {
        if (name_by_gid[i].empty()) continue;
        u32 r = root_of_user[i];
        if (name_of_root[r] == UINT32_MAX ||
            name_by_gid[i] < name_by_gid[name_of_root[r]])
            name_of_root[r] = (u32)i;
    }
    logstage("users linked");

    // ---- regroup event buckets
    std::vector< std::vector< std::pair<u32, const std::vector<ERec> *> > > ebm(NB);
    for (size_t seq = 0; seq < n_echunks; seq++) {
        auto &bl = ebuckets[seq];
        for (size_t b = 0; b < NB; b++) {
            if (!bl[b].empty()) ebm[b].emplace_back((u32)seq, &bl[b]);
        }
    }
    std::vector< std::vector< std::pair<u32, const std::vector<CRec> *> > > cbm(NB);
    for (size_t seq = 0; seq < n_cchunks; seq++) {
        auto &bl = cbuckets[seq];
        for (size_t b = 0; b < NB; b++) {
            if (!bl[b].empty()) cbm[b].emplace_back((u32)seq, &bl[b]);
        }
    }

    // ---- dedup events (parallel per bucket) -> ev2
    // thread-local generation-tagged arenas, sized to the largest bucket
    size_t emaxb = 0;
    for (size_t b = 0; b < NB; b++) {
        size_t tot = 0;
        for (auto &cp : ebm[b]) tot += cp.second->size();
        emaxb = std::max(emaxb, tot);
    }
    std::vector< std::vector<Ev2> > ev2_parts(NB);
    parallel_for(NB, 1, [&](size_t a, size_t b) {
        static thread_local EDedup ed;
        static thread_local size_t ed_cap = 0;
        if (ed_cap != emaxb) {
            ed.init(emaxb);
            ed_cap = emaxb;
        }
        for (size_t k = a; k < b; k++)
            dedup_events_bucket(ebm[k], esides, root_of_user, uid_empty_val,
                                known_products, unknown_rank, ws, we,
                                ev2_parts[k], ed);
    });
    std::vector<Ev2> ev2;
    {
        size_t total = 0;
        for (auto &p : ev2_parts) total += p.size();
        ev2.reserve(total);
        for (auto &p : ev2_parts) ev2.insert(ev2.end(), p.begin(), p.end());
        std::vector< std::vector<Ev2> >().swap(ev2_parts);
    }
    std::vector< std::vector< std::vector<ERec> > >().swap(ebuckets);
    logstage("events deduped");

    // ---- partition ev2 by root; per-bin: sort by (root, tsrel) + sessionize
    u32 maxroot = 0;
    for (size_t i = 0; i < ev2.size(); i++)
        maxroot = std::max(maxroot, (u32)(ev2[i].key >> 31));
    int rbits = 1;
    while (rbits < 31 && ((u64)1 << rbits) <= (u64)maxroot + 1) rbits++;
    int PBITS = rbits > 10 ? rbits - 10 : 0;
    size_t nbins = (size_t)1 << PBITS;
    int rootshift = rbits - PBITS;
    size_t T = std::min<size_t>((size_t)NTHREADS, 32);
    if (T == 0) T = 1;
    std::vector<size_t> boff(nbins + 1, 0);
    {
        // per-thread per-bin counts, then prefix, then parallel scatter
        std::vector<std::vector<u32>> hists(T);
        parallel_for(T, 1, [&](size_t a, size_t b) {
            for (size_t t = a; t < b; t++) {
                hists[t].assign(nbins, 0u);
                size_t lo = ev2.size() * t / T, hi2 = ev2.size() * (t + 1) / T;
                for (size_t i = lo; i < hi2; i++)
                    hists[t][(ev2[i].key >> 31) >> rootshift]++;
            }
        });
        size_t run = 0;
        std::vector<u32> tbase((size_t)T * nbins, 0u);
        for (size_t bn = 0; bn < nbins; bn++) {
            boff[bn] = run;
            for (size_t t = 0; t < T; t++) {
                tbase[(size_t)t * nbins + bn] = (u32)run;
                run += hists[t][bn];
            }
        }
        boff[nbins] = run;
        std::vector<Ev2> ev3(ev2.size());
        parallel_for(T, 1, [&](size_t a, size_t b) {
            for (size_t t = a; t < b; t++) {
                u32 *mybase = &tbase[(size_t)t * nbins];
                size_t lo = ev2.size() * t / T, hi2 = ev2.size() * (t + 1) / T;
                for (size_t i = lo; i < hi2; i++) {
                    size_t bn = (ev2[i].key >> 31) >> rootshift;
                    ev3[mybase[bn]++] = ev2[i];
                }
            }
        });
        ev2.swap(ev3);
    }
    std::vector< std::vector<Sess> > sess_parts(nbins);
    std::vector< std::vector<UDRow> > ud_parts(nbins);
    parallel_for(nbins, 1, [&](size_t a, size_t b) {
        for (size_t bn = a; bn < b; bn++) {
            size_t lo = boff[bn], hi2 = boff[bn + 1];
            size_t m = hi2 - lo;
            if (m == 0) continue;
            Ev2 *seg = ev2.data() + lo;
            std::sort(seg, seg + m,
                      [](const Ev2 &x, const Ev2 &y) { return x.key < y.key; });
            sessionize_group(seg, m, ws, organic, sess_parts[bn],
                             ud_parts[bn]);
        }
    });
    std::vector<Sess> sessions;
    std::vector<UDRow> ud_rows;
    {
        size_t ts_tot = 0, ud_tot = 0;
        for (auto &sp : sess_parts) ts_tot += sp.size();
        for (auto &up : ud_parts) ud_tot += up.size();
        sessions.reserve(ts_tot);
        ud_rows.reserve(ud_tot);
        for (auto &sp : sess_parts)
            sessions.insert(sessions.end(), sp.begin(), sp.end());
        for (auto &up : ud_parts)
            ud_rows.insert(ud_rows.end(), up.begin(), up.end());
    }
    std::vector< std::vector<Sess> >().swap(sess_parts);
    std::vector< std::vector<UDRow> >().swap(ud_parts);
    std::vector<Ev2>().swap(ev2);
    logstage("sessions built");

    // ---- click dedup (parallel per bucket) -> sort -> attribution
    size_t cmaxb = 0;
    for (size_t b = 0; b < NB; b++) {
        size_t tot = 0;
        for (auto &cp : cbm[b]) tot += cp.second->size();
        cmaxb = std::max(cmaxb, tot);
    }
    std::vector< std::vector<CFin> > cfin_parts(NB);
    parallel_for(NB, 1, [&](size_t a, size_t b) {
        static thread_local CDedup cd;
        static thread_local size_t cd_cap = 0;
        if (cd_cap != cmaxb) {
            cd.init(cmaxb);
            cd_cap = cmaxb;
        }
        for (size_t k = a; k < b; k++)
            dedup_clicks_bucket(cbm[k], root_of_user, uid_empty_val, n_camp,
                                ws, we, cfin_parts[k], cd);
    });
    std::vector< std::vector< std::vector<CRec> > >().swap(cbuckets);
    std::vector<CFin> clicks;
    {
        size_t total = 0;
        for (auto &p : cfin_parts) total += p.size();
        clicks.reserve(total);
        for (auto &p : cfin_parts) clicks.insert(clicks.end(), p.begin(), p.end());
        std::vector< std::vector<CFin> >().swap(cfin_parts);
    }
    {
        i64 click_base = ws - SEVEN;
        std::vector<u64> ckeys(clicks.size());
        for (size_t i = 0; i < clicks.size(); i++)
            ckeys[i] = ((u64)clicks[i].root << 31) |
                       (u64)((clicks[i].ts - click_base) & 0x7FFFFFFFull);
        parallel_radix_sort(ckeys, clicks);
        // tie-fix: equal (root, ts) runs sorted by click_id desc
        size_t i = 0;
        while (i < clicks.size()) {
            size_t j = i + 1;
            while (j < clicks.size() && clicks[j].root == clicks[i].root &&
                   clicks[j].ts == clicks[i].ts)
                j++;
            if (j - i > 1) {
                std::sort(clicks.begin() + i, clicks.begin() + j,
                          [&](const CFin &x, const CFin &y) {
                              const u8 *xb = &cl_arenas[x.arena][x.off];
                              const u8 *yb = &cl_arenas[y.arena][y.off];
                              size_t mn = std::min(x.len, y.len);
                              int c = memcmp(xb, yb, mn);
                              if (c != 0) return c > 0; // desc
                              return x.len > y.len;
                          });
            }
            i = j;
        }
    }
    logstage("clicks sorted");

    // ---- attribution (parallel groups over sessions)
    std::vector< std::pair<size_t, size_t> > sess_spans;
    {
        size_t i = 0;
        while (i < sessions.size()) {
            size_t j = i;
            while (j < sessions.size() && sessions[j].root == sessions[i].root)
                j++;
            sess_spans.emplace_back(i, j);
            i = j;
        }
    }
    std::vector< std::pair<u32, size_t> > c_ranges;
    for (size_t i = 0; i < clicks.size(); i++) {
        if (c_ranges.empty() || c_ranges.back().first != clicks[i].root)
            c_ranges.emplace_back(clicks[i].root, i);
    }
    // Tiles of session spans, one Agg per tile (not per root: a per-root Agg
    // costs ncamp1*ndays*32B and there can be ~100k roots).
    size_t n_tiles = std::min<size_t>(sess_spans.size(), (size_t)NTHREADS * 4);
    if (n_tiles == 0) n_tiles = 1;
    std::vector<size_t> tbounds;
    {
        size_t per = (sess_spans.size() + n_tiles - 1) / n_tiles;
        for (size_t i = 0; i < sess_spans.size(); i += per)
            tbounds.push_back(i);
        tbounds.push_back(sess_spans.size());
    }
    std::vector<Agg> aggs(tbounds.size() - 1, Agg(ncamp1, ndays));
    parallel_for(tbounds.size() - 1, 1, [&](size_t a, size_t b) {
        for (size_t t = a; t < b; t++) {
            Agg &tagg = aggs[t];
            std::vector<u32> counts(ncamp1, 0);
            std::vector<u8> in_active(ncamp1, 0);
            std::vector<u32> seen_gen(ncamp1, UINT32_MAX);
            std::vector<u16> active;
            for (size_t g = tbounds[t]; g < tbounds[t + 1]; g++) {
                size_t slo = sess_spans[g].first, shi = sess_spans[g].second;
                u32 root = sessions[slo].root;
                // binary search the root's click range
                size_t lo = 0, hi = c_ranges.size();
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if (c_ranges[mid].first < root) lo = mid + 1;
                    else hi = mid;
                }
                size_t c_lo = clicks.size(), c_hi = clicks.size();
                if (lo < c_ranges.size() && c_ranges[lo].first == root) {
                    c_lo = c_ranges[lo].second;
                    c_hi = (lo + 1 < c_ranges.size()) ? c_ranges[lo + 1].second
                                                      : clicks.size();
                }
                attribute_group(sessions.data() + slo, shi - slo, clicks.data(),
                                c_lo, c_hi, organic, ncamp1, ndays, ws_day,
                                (u32)slo, tagg, counts, in_active, seen_gen,
                                active);
            }
        }
    });
    Agg agg(ncamp1, ndays);
    for (auto &a : aggs) agg.merge(a);
    std::vector<Agg>().swap(aggs);
    logstage("attribution done");

    // ---- write outputs
    std::vector<BoxStr> camp_ids_all(ncamp1);
    std::vector<BoxStr> camp_ch_all(ncamp1);
    for (size_t c = 0; c < n_camp; c++) {
        camp_ids_all[c] = camp_ids[c];
        camp_ch_all[c] = camp_channels[c];
    }
    {
        BoxStr o{'o', 'r', 'g', 'a', 'n', 'i', 'c'};
        camp_ids_all[n_camp] = o;
        camp_ch_all[n_camp] = o;
    }

    mkdirs(out_dir);
    std::string ud = out_dir + "/user_daily";
    std::string cd = out_dir + "/campaign_daily";
    mkdirs(ud);
    mkdirs(cd);
    write_user_daily_parquet(ud + "/part-00000.parquet", ud_rows, name_by_gid,
                             name_of_root, categories);
    write_campaign_daily_parquet(cd + "/part-00000.parquet", camp_ids_all,
                                 camp_ch_all, agg.cred, agg.purc, agg.rev,
                                 agg.assi, ncamp1, ndays, ws_day);
    logstage("done");
    return 0;
}
