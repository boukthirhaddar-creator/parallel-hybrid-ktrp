/* Resolution exacte du K-TRP par programmation dynamique sur les sous-ensembles.
   Variante circuit (equation 2) : poids m+1 ... 1, retour compris.
   Variante chemin (equation 1)  : poids m ... 1, sans retour.
   f[S][i] : cout minimal d'un suffixe de tournee visitant exactement S et commencant en i.
   Les poids d'un suffixe ne dependent que de sa taille, ce qui rend la recurrence exacte.
   c[S]    : cout minimal d'une tournee visitant exactement S.
   g_k[S]  : cout minimal d'une partition de S en k tournees non vides.
   Usage : ktrp_exact fichier circuit|chemin                                          */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef double val;
#define INF 1e18
int main(int argc, char **argv) {
    FILE *fp = fopen(argv[1], "r"); int circuit = strcmp(argv[2], "circuit") == 0;
    int K, N; if (fscanf(fp, "%d %d", &K, &N) != 2) return 1;
    double *D = malloc(sizeof(double) * N * N);
    for (int i = 0; i < N * N; i++) if (fscanf(fp, "%lf", &D[i]) != 1) return 1;
    fclose(fp);
    int m = N - 1; uint32_t full = (1u << m) - 1, nS = 1u << m;
    val *f = malloc(sizeof(val) * (size_t)nS * m); val *c = malloc(sizeof(val) * nS);
    if (!f || !c) { fprintf(stderr, "memoire\n"); return 2; }
    for (size_t x = 0; x < (size_t)nS * m; x++) f[x] = INF;
    for (int i = 0; i < m; i++) f[(size_t)(1u << i) * m + i] = circuit ? D[(i + 1) * N + 0] : 0.0;
    for (uint32_t S = 1; S < nS; S++) {
        int s = __builtin_popcount(S); if (s < 2) continue;
        val w = circuit ? s : s - 1;      /* poids de l'arc quittant le premier client du suffixe */
        for (int i = 0; i < m; i++) if (S >> i & 1) {
            uint32_t R = S & ~(1u << i); val best = INF;
            for (int j = 0; j < m; j++) if (R >> j & 1) {
                val v = w * D[(i + 1) * N + (j + 1)] + f[(size_t)R * m + j];
                if (v < best) best = v;
            }
            f[(size_t)S * m + i] = best;
        }
    }
    for (uint32_t S = 1; S < nS; S++) {
        int s = __builtin_popcount(S); val w0 = circuit ? s + 1 : s, best = INF;
        for (int i = 0; i < m; i++) if (S >> i & 1) {
            val v = w0 * D[0 * N + (i + 1)] + f[(size_t)S * m + i]; if (v < best) best = v;
        }
        c[S] = best;
    }
    /* partition en K tournees non vides */
    val *g = malloc(sizeof(val) * nS), *h = malloc(sizeof(val) * nS);
    uint32_t **choix = malloc(sizeof(uint32_t *) * (K + 1));
    for (uint32_t S = 0; S < nS; S++) g[S] = c[S];
    for (int k = 2; k <= K; k++) {
        choix[k] = malloc(sizeof(uint32_t) * nS);
        for (uint32_t S = 0; S < nS; S++) {
            h[S] = INF; choix[k][S] = 0;
            if (__builtin_popcount(S) < k) continue;
            if (k == K && S != full) continue;       /* dernier niveau : ensemble complet seulement */
            uint32_t low = S & -S, rest = S ^ low;
            for (uint32_t r = rest;; r = (r - 1) & rest) {   /* T = low | r, T != S */
                uint32_t T = low | r;
                if (T != S) { val v = c[T] + g[S ^ T]; if (v < h[S]) { h[S] = v; choix[k][S] = T; } }
                if (r == 0) break;
            }
        }
        val *t = g; g = h; h = t;
    }
    printf("OPTIMUM %.6f\n", g[full]);
    /* reconstitution des tournees */
    uint32_t S = full;
    for (int k = K; k >= 1; k--) {
        uint32_t T = (k == 1) ? S : choix[k][S];
        /* sequence optimale de T */
        int s = __builtin_popcount(T); val w0 = circuit ? s + 1 : s; int first = -1; val best = INF;
        for (int i = 0; i < m; i++) if (T >> i & 1) { val v = w0 * D[i + 1] + f[(size_t)T * m + i]; if (v < best - 1e-9) { best = v; first = i; } }
        printf("TOURNEE 0"); uint32_t R = T; int cur = first;
        while (1) {
            printf(" %d", cur + 1); uint32_t R2 = R & ~(1u << cur); if (!R2) break;
            int ss = __builtin_popcount(R); val w = circuit ? ss : ss - 1; int nxt = -1;
            for (int j = 0; j < m; j++) if (R2 >> j & 1) {
                val v = w * D[(cur + 1) * N + (j + 1)] + f[(size_t)R2 * m + j];
                if (v <= f[(size_t)R * m + cur] + 1e-9) { nxt = j; break; }
            }
            R = R2; cur = nxt;
        }
        printf(" 0\n"); S ^= T;
    }
    return 0;
}
