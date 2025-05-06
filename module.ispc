// part1
export void computeQKt_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform float QK_t[],
    uniform const int b,
    uniform const int h,
    uniform const int H,
    uniform const int N,
    uniform const int d
){
    for (uniform int i = 0; i < N; i++) {
        for (uniform int j = 0; j < N; j++) {
            float val = 0;
            foreach (kid = 0 ... d){
                float val1 = Q[b * H * N * d + h * N * d + i * d + kid];
                float val2 = K[b * H * N * d + h * N * d + j * d + kid];
                val += val1 * val2;
            }
            uniform float sum = reduce_add(val);
            if (programIndex == 0) {
                QK_t[i * N + j] = sum;
            }
        }
    }
}
export void computeSoftmax_ispc(
    uniform float QK_t[],
    uniform const int N
){
    for (uniform int i = 0; i < N; i++) {
        float sum = 0;
        foreach (jid = 0 ... N) {
            float val = QK_t[i * N + jid];
            float exp_val = exp(val);
            QK_t[i * N + jid] = exp_val;
            sum += exp_val;
        }
        uniform float sum1 = reduce_add(sum);
        foreach (jid = 0 ... N) {
            if (jid < N) {
                float val = QK_t[i * N + jid];
                val /= sum1;
                QK_t[i * N + jid] = val;
            }   
        }
    }
}
    
export void matMul_PVO_ispc(
    uniform float QK_t[],
    uniform float V[],
    uniform float O[],
    uniform int b,
    uniform int h,
    uniform int H,
    uniform int N,
    uniform int d
){
    for (uniform int i = 0; i < N; i++) {

        for (uniform int j = 0; j < d; j++) {
            float sum = 0;
            foreach(kid = 0 ... N) {
                float val1 = QK_t[i * N + kid];
                #pragma ignore warning(perf)
                float val2 = V[b * H * N * d + h * N * d + kid * d + j];
                sum += val1 * val2;
            }
            uniform float sum1 = reduce_add(sum);
            O[b * H * N * d + h * N * d + i * d + j] = sum1;
        }
    }
}

// part2
export void computeQKt_blocked_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform float QK_t[],
    uniform const int b,
    uniform const int h,
    uniform const int H,
    uniform const int N,
    uniform const int d,
    uniform const int L
){
    for (uniform int i = 0; i < N; i += L) {

        for (uniform int j = 0; j < N; j += L) {
            
            for (uniform int k = 0; k < d; k += L) {

                for (uniform int i1 = i; i1 < min(i + L, N); i1++) {
                    
                    for (uniform int j1 = j; j1 < min(j + L, N); j1++) {
                        float sum = 0;
                        foreach (k1 = k ... min(k + L, d)) {
                            float val1 = Q[b * H * N * d + h * N * d + i1 * d + k1];
                            float val2 = K[b * H * N * d + h * N * d + j1 * d + k1];
                            sum += val1 * val2;
                        }
                        uniform float sum1 = reduce_add(sum);
                        uniform float sum2 = QK_t[i1 * N + j1];
                        QK_t[i1 * N + j1] = sum1 + sum2;
                    }

                }

            }


        }
    }
}
export void matMul_PVO_blocked_ispc(
    uniform float QK_t[],
    uniform float V[],
    uniform float O[],
    uniform int b,
    uniform int h,
    uniform int H,
    uniform int N,
    uniform int d,
    uniform int L
){
    for (uniform int i = 0; i < N; i += L) {

        for (uniform int j = 0; j < d; j += L) {

            for (uniform int k = 0; k < N; k += L) {

                for (uniform int i1 = i; i1 < min(i + L, N); i1++) {
                    
                    for (uniform int j1 = j; j1 < min(j + L, d); j1++) {
                        float val = 0;
                        foreach (k1 = k ... min(k + L, N)) {
                            float val1 = QK_t[i1 * N + k1];
                            #pragma ignore warning(perf)
                            float val2 = V[b * H * N * d + h * N * d + k1 * d + j1];
                            val += val1 * val2;
                        }
                        uniform float sum1 = reduce_add(val);
                        uniform float sum2 = O[b * H * N * d + h * N * d + i1 * d + j1];
                        O[b * H * N * d + h * N * d + i1 * d + j1] = sum1 + sum2;
                    }

                }

            }

        }
    }
}

// part3
export void computePart3_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform const float V[],
    uniform float O[],
    uniform float ORow[],
    uniform const int b,
    uniform const int h,
    uniform const int H,
    uniform const int N,
    uniform const int d,
    uniform const int i
){
    uniform float exp_sum = 0;
    for (uniform int j = 0; j < N; j++) {
        float val = 0;
        foreach (k = 0 ... d) {
            float val1 = Q[b * H * N * d + h * N * d + i * d + k];
            float val2 = K[b * H * N * d + h * N * d + j * d + k];
            val += val1 * val2;
        }
        uniform float sum = reduce_add(val);
        uniform float exp_val = exp(sum);
        ORow[j] = exp_val;
        exp_sum += exp_val;
    }
    for (uniform int j = 0; j < d; j++) {
        float val = 0;
        foreach (k = 0 ... N) {
            float val1 = ORow[k];
            float val2 = V[b * H * N * d + h * N * d + k * d + j];
            val += val1 * val2;
        }
        uniform float sum = reduce_add(val);
        sum /= exp_sum;
        O[b * H * N * d + h * N * d + i * d + j] = sum;
    }
}

// part4
export void computePart4_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform const float V[],
    uniform float O[],
    uniform float Sij[],
    uniform float Pij[],
    uniform float Kj[],
    uniform float Vj[],
    uniform float Qi[],
    uniform float Oi[],
    uniform float l[],
    uniform float li[],
    uniform float lij[],
    uniform float lnew[],
    uniform const int b,
    uniform const int h,
    uniform const int H,
    uniform const int N,
    uniform const int d,
    uniform const int i,
    uniform const int j,
    uniform const int Br,
    uniform const int Bc
){
    // Sij = QiKj_t, Pij = exp(Sij), Lij = rowsum(Pij), Lnew = Li + Lij
    for (uniform int i1 = i; i1 < min(N, i + Br); i1++) {
        uniform float exp_sum = 0;
        uniform int i_local = i1 - i;
        for (uniform int j1 = j; j1 < min(N, j + Bc); j1++) {
            float val = 0;
            uniform int j_local = j1 - j;
            foreach(k = 0 ... d) {
                float val1 = Qi[i_local * d + k];
                float val2 = Kj[j_local * d + k];
                val += val1 * val2;
            }
            uniform float sum = reduce_add(val);
            Sij[i_local * Bc + j_local] = sum;
            uniform float exp_val = exp(sum);
            Pij[i_local * Bc + j_local] = exp_val;
            exp_sum += exp_val;
        }
        lij[i_local] = exp_sum;
        lnew[i_local] = li[i_local] + lij[i_local];
    }
    // Oi <- (liOi + PijVj) / lnew
    for (uniform int i1 = i; i1 < min(N, i + Br); i1++) {
        uniform int i_local = i1 - i;
        for (uniform int k = 0; k < d; k++) {
            float val = 0;
            foreach (j1 = j ... min(N, j + Bc)) {
                int j_local = j1 - j;
                float val1 = Pij[i_local * Bc + j_local];
                float val2 = Vj[j_local * d + k];
                val += val1 * val2;
            }
            uniform float val3 = Oi[i_local * d + k];
            uniform float val_u = reduce_add(val);
            val_u += val3 * li[i_local]; 
            val_u /= lnew[i_local];
            Oi[i_local * d + k] = val_u;
        }
    }
    // Write Oi, lnew to O and L;
    for (int i1 = i; i1 < min(N, i + Br); i1++) {
        int i_local = i1 - i;
        l[i1] = lnew[i_local];
        foreach (k = 0 ... d) {
            float val = Oi[i_local * d + k];
            O[b * H * N * d + h * N * d + i1 * d + k] = val;
        }
    }
}