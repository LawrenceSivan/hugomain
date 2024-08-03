+++
title = 'Test'
date = 2024-08-02T22:33:17+08:00
+++

TEST

## P4396 [AHOI2013] 作业

莫队+值域分块

题意：给出一个静态区间和多个询问，询问分为下面两种：

- 在区间  $[l,r]$ 中有多少个 $i$，使得 $a_i∈[a,b]$；
- 在区间  $[l,r]$ 中有多少种 $a_i(i∈[l,r])$，使得 $ai∈[a,b]$

第二问其实就是第一问去重后的结果，统计时可以直接去重，只需要考虑第一问，给定区间数颜色，想到莫队，但是还有个值域的限制，我们可以想到前缀和相减，直接上个树状数组带 $\log$ ,复杂度是 $O(n \sqrt n\log n)$，考虑值域分块，即可做到 $O(1)$ 插入，$O(\sqrt n)$ 查询。块长取到 $\dfrac {n}{\sqrt m}$ 为最优，最终复杂度 $O（n\sqrt m+m\sqrt n）$

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e5 + 5;

int n, m, block, lal = 1, lar = 0;

int a[maxn], belong[maxn], cnt[maxn];

int ans1[maxn], ans2[maxn], val1[maxn], val2[maxn];

struct node {
    int l, r, a, b, id;
    bool operator < (const node &b)const {
        return belong[l] != belong[b.l] ? l < b.l : r < b.r;
    }
} q[maxn];

inline void add(int x) {
    if (!cnt[x])val2[belong[x]]++;

    cnt[x]++;
    val1[belong[x]]++;
}

inline void del(int x) {
    cnt[x]--;

    if (!cnt[x])val2[belong[x]]--;

    val1[belong[x]]--;
}

inline int query1(int a, int b) {
    int res = 0;

    if (belong[a] == belong[b]) {
        for (int i = a; i <= b; i++)res += cnt[i];

        return res;
    }

    for (int i = a; i <= block * belong[a]; i++)res += cnt[i];

    for (int i = belong[a] + 1; i < belong[b]; i++)res += val1[i];

    for (int i = block * (belong[b] - 1) + 1; i <= b; i++)res += cnt[i];

    return res;
}


inline int query2(int a, int b) {
    int res = 0;

    if (belong[a] == belong[b]) {
        for (int i = a; i <= b; i++)res += cnt[i]>0;

        return res;
    }

    for (int i = a; i <= block * belong[a]; i++)res += cnt[i]>0;

    for (int i = belong[a] + 1; i < belong[b]; i++)res += val2[i];

    for (int i = block * (belong[b] - 1) + 1; i <= b; i++)res += cnt[i]>0;

    return res;
}

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(n), read(m);
    block = 1.0 * n / sqrt(m);

    for (int i = 1; i <= n; i++) {
        read(a[i]);
        belong[i] = (i - 1) / block + 1;
    }

    for (int i = 1; i <= m; i++) {
        read(q[i].l), read(q[i].r), read(q[i].a), read(q[i].b);
        q[i].id = i;
    }

    sort(q + 1, q + 1 + m);

    for (int i = 1; i <= m; i++) {
        int l = q[i].l, r = q[i].r;

        while (l < lal)lal--, add(a[lal]);

        while (l > lal)del(a[lal]), lal++;

        while (r < lar)del(a[lar]), lar--;

        while (r > lar)lar++, add(a[lar]);

        ans1[q[i].id] = query1(q[i].a, q[i].b);
        ans2[q[i].id] = query2(q[i].a, q[i].b);
    }

    for (int i = 1; i <= m; i++) {
        printf("%d %d\n", ans1[i], ans2[i]);
    }


    return 0;
}
```

## CF1620E Replace the Numbers

维护一个数列，这个数列初始为空。

对于这个数列，总共有 $q$ 次操作，每次操作分为如下两个种类：

1. `1 x`，意为在数列末尾加一个数字
2. `2 x y`，意为将数列中所有值为 $x$ 的数的值替换成 $y$

请在 $q$ 次操作后，输出这个数列。$x,y,z\le5\times 10^5$

---

同样的数字批量修改，考虑并查集。记录每个位置 $i$ 的数字是多少 $val[i]$，值为 $x$ 的位置有哪些（通过并查集，事实上只记录路径压缩后的根节点即可），碰到相同的数字直接拉在一起，否则初始化根节点为自身，修改的时候如果 $y$ 已经存在，只需要把值为 $x$ 的节点全都拉到值为 $y$ 的根节点下，如果 $y$ 不存在，把 $x$ 的所有信息继承给 $y$ 即可。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 5e5 + 5;

int n, q, a[maxn];

int f[maxn], val[maxn], pos[maxn];

int find(int x) {
    return x == f[x] ? x : f[x] = find(f[x]);
}

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(q);

    for (int i = 1, op, x, y; i <= q; i++) {
        read(op), read(x);

        if (op == 1) {
            a[++n] = i, f[i] = i, val[i] = x;

            if (pos[x])f[i] = pos[x];
            else pos[x] = i;
        } else {
            read(y);

            if (x != y && pos[x]) {
                if (pos[y]) {
                    f[pos[x]] = pos[y];
                    pos[x] = 0;
                } else {
                    pos[y] = pos[x];
                    val[pos[x]] = y;
                    pos[x] = 0;
                }
            }

        }
    }

    for (int i = 1; i <= n; i++)printf("%d ", val[find(a[i])]);


    return 0;
}
```

## P3302 [SDOI2013] 森林

- 查询 $x$ 到 $y$ 路径第 $K$ 小
- 在 $x$ 与 $y$ 中间连一条边

树上主席树+启发式合并

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 8e4 + 5;

int testcase;

int n, m, T, lastans, nn, tot;

int lg[maxn], w[maxn], a[maxn], core[maxn], root[maxn];

int head[maxn], to[maxn << 1], nxt[maxn << 1], cnt;

inline void add(int u, int v) {
    nxt[++cnt] = head[u];
    to[cnt] = v;
    head[u] = cnt;
}

int f[maxn][20], dep[maxn], size[maxn];

struct node {
    int lc, rc, v;
#define ls(x) st[(x)].lc
#define rs(x) st[(x)].rc
#define val(x) st[(x)].v
} st[maxn * 400];

void modify(int pre, int &rt, int l, int r, int wal) {
    rt = ++tot;
    st[rt] = st[pre], val(rt)++;

    if (l == r)return;

    int mid = (l + r) >> 1;

    if (wal <= mid)modify(ls(pre), ls(rt), l, mid, wal);
    else modify(rs(pre), rs(rt), mid + 1, r, wal);
}

void dfs(int u, int fa, int rt) {
    f[u][0] = fa, dep[u] = dep[fa] + 1;
    size[rt]++, core[u] = rt;
    modify(root[fa], root[u], 1, nn, w[u]);

    for (int i = 1; i < 20; i++) {
        f[u][i] = f[f[u][i - 1]][i - 1];
    }

    for (int i = head[u]; i; i = nxt[i]) {
        int v = to[i];

        if (v == fa)continue;

        dfs(v, u, rt);
    }
}

inline int LCA(int x, int y) {
    if (dep[x] < dep[y])swap(x, y);

    while (dep[x] > dep[y])x = f[x][lg[dep[x] - dep[y]]];

    if (x == y)return x;

    for (int i = 19; ~i; i--) {
        if (f[x][i] != f[y][i])x = f[x][i], y = f[y][i];
    }

    return f[x][0];
}

int query(int x, int y, int z, int fz, int l, int r, int wal) {
    if (l == r)return l;

    int sum = val(ls(x)) + val(ls(y)) - val(ls(z)) - val(ls(fz));
    int mid = (l + r) >> 1;

    if (sum >= wal)return query(ls(x), ls(y), ls(z), ls(fz), l, mid, wal);
    else return query(rs(x), rs(y), rs(z), rs(fz), mid + 1, r, wal - sum);
}


inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(testcase);
    read(n), read(m), read(T);

    for (int i = 2; i < maxn; i++)lg[i] = lg[i >> 1] + 1;

    for (int i = 1; i <= n; i++)read(w[i]), a[i] = w[i], core[i] = i;

    sort(a + 1, a + 1 + n);
    nn = unique(a + 1, a + 1 + n) - a - 1;

    for (int i = 1; i <= n; i++) {
        w[i] = lower_bound(a + 1, a + 1 + nn, w[i]) - a;
    }

    for (int i = 1, u, v; i <= m; i++) {
        read(u), read(v);
        add(u, v);
        add(v, u);
    }

    for (int i = 1; i <= n; i++) {
        if (core[i] == i)dfs(i, 0, i);
    }

    for (int i = 1, x, y, k; i <= T; i++) {
        string op;
        cin >> op;
        read(x), read(y), x ^= lastans, y ^= lastans;

        if (op[0] == 'Q') {
            read(k), k ^= lastans;
            int z = LCA(x, y);
            lastans = a[query(root[x], root[y], root[z], root[f[z][0]], 1, nn, k)];
            printf("%d\n", lastans);
        } else {
            add(x, y), add(y, x);
            int cox = core[x], coy = core[y];

            if (size[cox] < size[coy])swap(x, y);

            dfs(y, x, cox);
        }
    }



    return 0;
}


```

## P5356 [Ynoi2017] 由乃打扑克

给你一个长为 $n$ 的序列 $a$，需要支持 $m$ 次操作，操作有两种：

- 查询区间 $[l,r]$ 的第 $k$ 小值。
- 区间 $[l,r]$ 加上 $k$。

动态区间 $k$ 小值，发现要支持区间修改操作。可惜树套树打不了标记。

考虑分块，对于每个块块内排序，区间加直接打个标记，散块暴力加，查询的时候二分一个值，全部小于等于二分值的块直接加进去统计数量（块内单调，比较端点），否则在块内 `upper_bound` ，散块依然暴力统计，最后都加起来和 $k$ 比较即可。

块长 $\sqrt n$ ,复杂度 $O(n\sqrt {n} \log n\log V)$

据说被卡掉了？交了一下发现没有。

在卡了一年之后发现 hack #1 一直过不去。怎么回事呢？打开评论区一看原来需要开 `long long`，感觉有点脑溢血了。

看着还行，实则调了很久。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e5 + 5;
#define int long long

int n, m, block, times = 1;

int a[maxn], belong[maxn], tag[maxn];

struct node {
    int v, id;
    bool operator < (const node &a)const {
        return v < a.v;
    }
} b[maxn];

inline int bl(int B) {
    return (B - 1) * block + 1;
}

inline int br(int B) {
    return B * block;
}

inline int Upper_bound(int l, int r, int val) {
    int L = l, R = r;

    while (L < R) {
        int mid = (L + R) >> 1;

        if (b[mid].v <= val)L = mid + 1;
        else R = mid;
    }

    return L;
}

inline int query(int l, int r, int k) {
    if (r - l + 1 < k)return -1;

    if (belong[l] == belong[r]) {
        int c[r - l + 2];

        for (int i = l; i <= r; i++) {
            c[i - l + 1] = a[i];
        }

        nth_element(c + 1, c + k, c + 1 + r - l + 1);
        return c[k] + tag[belong[l]];
    }

    int L = -2e4 * times, R = 2e4 * times;

    while (L < R) {
        int mid = (L + R) >> 1, cnt = 0;

        for (int i = belong[l] + 1; i < belong[r]; i++) {
            if (b[br(i)].v + tag[i] <= mid)cnt += block;
            else if (b[bl(i)].v + tag[i] <= mid) {
                cnt += Upper_bound(bl(i), br(i), mid - tag[i]) - bl(i);
            }
        }

        for (int i = l; i <= br(belong[l]); i++) {
            if (a[i] + tag[belong[l]] <= mid)cnt++;
        }

        for (int i = bl(belong[r]); i <= r; i++) {
            if (a[i] + tag[belong[r]] <= mid)cnt++;
        }

        if (cnt >= k)R = mid;
        else L = mid + 1;
    }

    return L;
}

inline void update(int l, int r, int k) {
    for (int i = bl(belong[l]); i <= min(n, br(belong[l])); i++) {
        if (b[i].id >= l && b[i].id <= r) {
            b[i].v += k, a[b[i].id] += k;
        }
    }

    sort(b + bl(belong[l]), b + 1 + min(br(belong[l]), n));
}

inline void modify(int l, int r, int k) {
    if (belong[l] == belong[r]) {
        update(l, r, k);
        return;
    }

    for (int i = belong[l] + 1; i < belong[r]; i++)tag[i] += k;

    update(l, br(belong[l]), k), update(bl(belong[r]), r, k);
}

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

signed main() {
    read(n), read(m);

    block = sqrt(n);

    for (int i = 1; i <= n; i++) {
        read(a[i]), b[i].v = a[i], b[i].id = i;
        belong[i] = (i - 1) / block + 1;
    }

    for (int i = 1; i <= n; i += block) {
        sort(b + bl(belong[i]) , b + min(br(belong[i]), n) + 1);
    }

    sort(b + br(n / block) + 1, b + 1 + n);

    for (int i = 1, op, l, r, k; i <= m; i++) {
        read(op), read(l), read(r), read(k);

        if (op == 1) {
            printf("%d\n", query(l, r, k));
        } else {
            times++;
            modify(l, r, k);
        }
    }

    return 0;
}
```

然后发现题解有 $O(n\sqrt{n}\log V)$ 的,块长取 $S=\sqrt n\log n$,每次重构的时候,考虑到加的部分是有序的,不加的部分也是有序的,实际上不需要排序,直接归并即可把重构复杂度从 $O(S\log S)$ 变成 $O(S)$,这样单次修改是 $O(\sqrt n\log n)$,查询的时候块数是 $O(n/S)$,也就是 $O(\dfrac{\sqrt n}{\log n})$,查询复杂度 $O(\sqrt n\log V)$

## P7962 [NOIP2021] 方差

补补当年的遗憾。

给定长度为 $n$ 的非严格递增正整数数列 $$1 \le a_1 \le a_2 \le \cdots \le a_n $$，每次可以任意选择一个 $a_i$ 变为 $$a_{i - 1} + a_{i + 1} - a_i$$。求在若干次操作之后，该数列的方差最小值是多少。请输出最小值乘以 $n^2$ 的结果。

---

首先拆一拆方差式子


$$
D = \frac{ \sum_\limits{i = 1}^{n} {(a_i - \bar a)}^2}{n}=\dfrac{\sum_\limits{i = 1}^{n}(a_i^2-2a_i\bar a+\bar a^2)}{n}=\dfrac{\sum_\limits{i = 1}^{n}a_i^2-2\bar a\sum\limits _{i=1}^{n}a_i+n\bar a^2}{n}
$$

$$
=\dfrac{\sum_\limits{i = 1}^{n}a_i^2-2n\bar a^2+n\bar a^2}{n}=\dfrac{\sum_\limits{i = 1}^{n}a_i^2-n\bar a^2}{n}
$$

也就是均值的平方减去平方的均值，最后只需要求

$$
n\sum_\limits{i = 1}^{n}a_i^2-\sum\limits_{i=1}^{n}a_i
$$
观察题目中的操作，可以发现对于 $a_i$ 的操作，事实上等价于对差分数组 $d_i$ 和 $d_{i+1}$ 进行交换。多轮操作后其实就相当于差分数组除了第一位剩下的可以随便重排。

之后有一个性质，需要尽量保证差分数组呈现单谷。证明考虑微扰。假设现在满足单谷，对于谷的右侧或者左侧，交换相邻两个，会使得当前位置 $a_i$ 变大，而前面和后面都不变，也就是从 $a_i$ 开始，后面的数字统一上升了一层，导致差距变大，方差变大。

然后做到这可以随机化，可以暴搜枚举每一个数字放在什么位置，似乎大几十分就到手了。(好像真有人直接搜满了，有点那啥了)

考虑 DP，先对差分数组排序，然后一个一个插入。

回到答案的式子

$$
n\sum_\limits{i = 1}^{n}a_i^2-\sum\limits_{i=1}^{n}a_i
$$
设 $f[i][j]$ 表示考虑了前 $i$ 个位置的 $d$ ，$\sum a_i=j$ 时，$\sum a_i^2$ 的最小值。

当插入右边的时候：记 $t=\sum d_i$，则 $a[i]=t$

$f[i][j]=min(f[i-1][j],f[i-1][j-a[i]]+a[i]^2)$ 

当插入左边的时候：后面所有位置都会加上 $d_i$ ，也就是总和加上了 $i\times d_i$ ，平方和加上了 


$$
\sum(x_k+d_i)^2-\sum x_k^2=\sum2d_ix_k+\sum d_i^2\\=2d_i\sum x_k+i\times d_i^2=2d_i(j-i\times d_i)+i\times d_i^2
$$

$f[i][j]=min(f[i-1][j],f[i-1][j-i\times d_i]+2d_i(j-i\times d_i)+i\times d_i^2)$

显然可以滚动数组，复杂度 $O(n^2V)$

然后发现 #23-#25 $n$ 大的离谱，究竟是怎么一回事呢？然后发现 $a_i$ 小的奇怪，这也就意味着 $a$ 数组有大量重复，差分是 0，那我们直接无脑往中间放就好了，因为题目保证数组单调不减。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e4 + 5;

int n, m, now, t, sum, a[maxn], d[maxn];

using i64 = long long;
constexpr i64 INF = 0x3f3f3f3f3f3f3f3f;

i64 f[2][maxn * 600], ans = INF;

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(n), m = n - 1;

    for (int i = 1; i <= n; i++) {
        read(a[i]);
    }

    sum = n * a[n];

    for (int i = 1; i <= m; i++)d[i] = a[i + 1] - a[i];

    for (int i = 1; i <= n; i++)sum += i * d[i];

    sort(d + 1, d + 1 + m);
    memset(f, 0x3f, sizeof(f));

    f[0][0] = 0;

    for (int i = 1; i <= m; i++) {
        if (d[i] == 0)continue;

        now ^= 1, t += d[i];

        for (int j = 0; j <= sum; j++) {
            f[now][j] = INF;

            if (j - t >= 0)
                f[now][j] = min(f[now][j], f[now ^ 1][j - t] + (i64)t * t);

            if (j - i * d[i] >= 0)
                f[now][j] = min(f[now][j],
                                f[now ^ 1][j - i * d[i]] + (i64)2 * d[i] * (j - i * d[i]) + (i64)i * d[i] * d[i]);
        }
    }

    for (int i = 0; i <= sum; i++) {
        if (f[now][i] != INF) {
            ans = min(ans, n * f[now][i] - (i64)i * i);
        }
    }

    printf("%lld\n", ans);


    return 0;
}
```

## P7961 [NOIP2021] 数列

给定整数 $n, m, k$，和一个长度为 $m + 1$ 的正整数数组 $v_0, v_1, \ldots, v_m$。

对于一个长度为 $n$，下标从 $1$ 开始且每个元素均不超过 $m$ 的非负整数序列 $$\{a_i\}$$ ，我们定义它的权值为 $v_{a_1} \times v_{a_2} \times \cdots \times v_{a_n}$。

当这样的序列 $\{a_i\}$ 满足整数 $S = 2^{a_1} + 2^{a_2} + \cdots + 2^{a_n}$ 的二进制表示中 $1$ 的个数不超过 $k$ 时，我们认为 $\{a_i\}$ 是一个合法序列。

计算所有合法序列 $\{a_i\}$ 的权值和对 $998244353$ 取模的结果。

 $1 \leq k \leq n \leq 30$，$0 \leq m \leq 100$，$1 \leq v_i < 998244353$

---

注意到合法序列的要求是 $S$ 的二进制表示中有多少个 $1$ ，而这个东西是涉及到进位的。从低位向高位考虑进位，设状态 $f(i,j,k,l)$ 表示当前考虑到前 $i$ 位，前 $i$ 位有 $j$ 个 $1$ ，填了 $k$ 个 $a$ ，向下一位进位 $p$，的合法序列的权值和

考虑这个状态怎么向下转移。对于下一位，如果有 $t$ 个 $a$ 可以对这位有 $1$ 的贡献，那么 再加上上一位的进位 这一位一共有 $t+l$ 个 $1$ ，同时这一位与此同时就确认了 $t$ 个 $a$ ，最后当前位置就是 $(t+p)\bmod 2$,向下一位的进位就是 $\lfloor \dfrac{t+p}{2}\rfloor$ .

因此，$f(i,j,k,l)$ 可以转移到 $f(i+1,j+(t+p)\bmod2,k+t,\lfloor \dfrac{t+p}{2}\rfloor)$

接下来考虑贡献。不难得出就是 $v_i^t\times \binom{n-k}{t}$

转移方程：

$$
f(i+1,j+(t+l)\bmod2,k+t,\lfloor \dfrac{t+l}{2}\rfloor)= \sum\limits_{t=0}^{n-k}f(i,j,k,l)\times v_i^t\times \binom{n-k}{t}
$$

预处理组合数,还有 $v$ 的次幂。

最后直到，到了第 $m$ 位依然是可以发生进位的，所以需要快速统计出进完位后第 $m$ 位往后的 $1$ 的个数，再加上 $j$ 这一维的数量，如果小于等于输入的 $k$，就可以统计进答案，统计答案只需要看 $m+1$。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 105;
const int mod = 998244353;

using i64 = long long;

int n, m, K, ans;

int C[35][35], vp[maxn][35];

int f[maxn][35][35][16];

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(n), read(m), read(K);

    for (int i = 0; i <= m; i++) {
        read(vp[i][1]), vp[i][0] = 1;

        for (int j = 2; j <= n; j++)
            vp[i][j] = ((i64)vp[i][j - 1] * vp[i][1]) % mod;
    }

    for (int i = 0; i <= n; i++)C[i][0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % mod;
        }
    }

    f[0][0][0][0] = 1;

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= K; j++) {
            for (int k = 0; k <= n; k++) {
                for (int l = 0; l <= (n >> 1); l++) {
                    for (int t = 0; t <= n - k; t++) {
                        (f[i + 1][j + ((t + l) & 1)][k + t][(t + l) >> 1] += (i64)f[i][j][k][l] * vp[i][t] % mod * C[
                                    n - k][t] % mod) %= mod;
                    }
                }
            }
        }
    }

    for (int j = 0; j <= K; j++) {
        for (int l = 0; l <= (n >> 1); l++) {
            if (j + __builtin_popcount(l) <= K) {
                ans = (ans + f[m + 1][j][n][l]) % mod;
            }
        }
    }

    printf("%d\n", ans);


    return 0;
}
```

## CF1242B 0-1 MST

给定一张完全图, 其中有 $m$ 条边权值为 $1$, 求这张图的最小生成树

$n,m\le10^5$

转化一下就是完全图，有 $m$ 对点之间没有连边，求出连通块个数 $-1$.

---

完全图共 $\dfrac{n\times (n-1)}{2}$ 条边，可以发现不连边的情况其实相较很少。也就是说，点之间连接的可能其实很大。我们考虑可能性最大的一个，也就是断边最少的那一个点，如果我们直接从这个点出发，把所有和他相连的点都和他合并，可以想象到这应该是一个很大的连通块。

我们理性考虑一下，相当于 $2m$ 个边的端点分给 $n$ 个点，连接 $1$ 边最少的点肯定连接了小于等于 $\dfrac{2m}{n}$ 条边,也就是这一个很大的连通块大小大概在 $n-\dfrac{2m}{n}$,剩下的点大概在 $\dfrac{2m}{n}$ ，之后剩下的这些单点和那个巨大的连通块暴力合并就可以了，这部分是 $O(n\times \dfrac{2m}{n})=O(m)$

神奇吧，这个东西居然是线性的!

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e5 + 5;
const int mod = 998244353;

using i64 = long long;

int n, m;

int f[maxn];

int find(int x) {
    return x == f[x] ? x : f[x] = find(f[x]);
}

int deg[maxn], block[maxn], vis[maxn];

inline void merge(int x, int y) {
    x = find(x), y = find(y);
    f[x] = y;
}

int head[maxn], to[maxn << 1], nxt[maxn << 1], cnt;

inline void add(int u, int v) {
    nxt[++cnt] = head[u];
    to[cnt] = v;
    head[u] = cnt;
}

vector <int> point;

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(n), read(m);

    for (int i = 1; i < maxn; i++)f[i] = i;

    for (int i = 1, u, v; i <= m; i++) {
        read(u), read(v);
        add(u, v), add(v, u);
        deg[u]++, deg[v]++;
    }

    int Min = deg[1], pos = 1;

    for (int i = 1; i <= n; i++) {
        if (deg[i] < Min)Min = deg[i], pos = i;
    }

    for (int i = head[pos]; i; i = nxt[i]) {
        int v = to[i];
        vis[v] = 1;
    }

    for (int i = 1; i <= n; i++) {
        if (!vis[i]) {
            merge(i, pos);
            block[i] = pos;
        }
    }

    memset(vis, 0, sizeof(vis));

    for (int i = 1; i <= n; i++) {
        if (!block[i])point.emplace_back(i);
    }

    for (int u = 1; u <= n; u++) {
        int tot = 0;

        for (int i = head[u]; i; i = nxt[i]) {
            int v = to[i];
            vis[v] = 1;
            tot += (block[v] == pos);
        }

        for (auto v : point) {
            if (vis[v])continue;

            merge(u, v);
        }

        if ((tot + point.size()) < n)merge(u, pos);

        for (int i = head[u]; i; i = nxt[i]) {
            int v = to[i];
            vis[v] = 0;
        }
    }

    int ans = 0;

    for (int i = 1; i <= n; i++) {
        ans += (i == f[i]);
    }

    printf("%d\n", ans - 1);


    return 0;
}
```

## P9869 [NOIP2023] 三值逻辑

一开始给定序列 $x$ ，每个位置都有一个初值（共有三种，T，F，U），接下来 $m$ 个操作，每个操作三种可能：

- 把 $x_i$ 置成 T,F,U 三者其一。
- $x_i \leftarrow x_j$
- $x_i \leftarrow \neg x_j $

---

一眼扩展域并查集。一个点拆成两个，一个代表正，一个代表反，修改操作直接考虑连边。判断时只需考虑是否某个点的正点和反点在同一个集合。然后兴冲冲写了一发 wa 了，仔细一考虑发现每次更改只是单点更改，并不是一条链，好像有点愚蠢了。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 1e5 + 5;
const int mod = 998244353;

using i64 = long long;

int testcase, T, n, m, cnt;

char c[3];

int f[maxn << 1], val[maxn];

inline void init() {
    for (int i = 1; i < (maxn << 1); i++)f[i] = i;

    for (int i = 1; i <= n; i++)val[i] = i;

    cnt = 0;
}

int find(int x) {
    return x == f[x] ? x : f[x] = find(f[x]);
}

inline void merge(int x, int y) {
    f[find(x)] = find(y);
}

inline void read(int &res) {
    int f = 1;
    res = 0;
    char ch = getchar();

    while (!isdigit(ch)) {if (ch == '-')f = -1; ch = getchar();}

    while (isdigit(ch)) {res = res * 10 + (ch ^ 48); ch = getchar();}

    res *= f;
}

int main() {
    read(testcase), read(T);

    while (T--) {
        read(n), read(m);
        init();

        //2n+1:T   2n+2:F   2n+3:U
        for (int i = 1, u, v; i <= m; i++) {
            scanf("%s", c);

            if (c[0] == '+') {
                read(u), read(v);
                val[u] = val[v];
            } else if (c[0] == '-') {
                read(u), read(v);
                val[u] = -val[v];
            } else {
                read(u);

                if (c[0] == 'T')val[u] = 2 * n + 1;

                if (c[0] == 'F')val[u] = 2 * n + 2;

                if (c[0] == 'U')val[u] = 2 * n + 3;
            }
        }

        for (int i = 1; i <= n; i++) {
            if (abs(val[i]) == 2 * n + 1 || abs(val[i]) == 2 * n + 2)continue;

            if (abs(val[i]) == 2 * n + 3)merge(i, n + i);

            else if (val[i] > 0) {
                merge(i, val[i]);
                merge(i + n, val[i] + n);
            } else if (val[i] < 0) {
                merge(i, -val[i] + n);
                merge(i + n, -val[i]);
            }
        }

        for (int i = 1; i <= n; i++) {
            if (find(i) == find(i + n))cnt++;
        }

        printf("%d\n", cnt);
    }
    
    return 0;
}
```

另一种做法是考虑拆点，某个位置被修改了 $k$ 次就拆成 $k+1$ 个点，操作一直接新建 $x$ 版本然后暴力赋值，操作二和操作三就是先新建版本，然后 $x$ 的新版本向 $y$ 的新版本连一条边，如果是取反边权就是 $1$，否则是 $0$ 。之后我们在每一个位置的最初版本和最后版本连一条边权为 $0$ 的边，代表二者相等。都处理完后就变成了一个基环树森林。之后基环树找环，判断基环树边权和是否是奇数，如果是，那么这一整颗树都是 $U$,或者是这一个树中有 $U$ ，那么直接整个树也是 $U$.

代码可能找个闲的时候再写。