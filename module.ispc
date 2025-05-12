// part1
export void part1_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform float QK_t[],
    uniform float V[],
    uniform float O[],
    uniform const int N,
    uniform const int d
){
    // QK^t
    for (uniform int i = 0; i < N; i++) {
        for (uniform int j = 0; j < N; j++) {
            float val = 0;
            foreach (k = 0 ... d){
                float val1 = Q[i * d + k];
                float val2 = K[j * d + k];
                val += val1 * val2;
            }
            uniform float sum = reduce_add(val);
            if (programIndex == 0) {
                QK_t[i * N + j] = sum;
            }
        }
    }
    // softmax(QK^t)
    for (uniform int i = 0; i < N; i++) {
        float sum = 0;
        foreach (j = 0 ... N) {
            float val = QK_t[i * N + j];
            float exp_val = exp(val);
            QK_t[i * N + j] = exp_val;
            sum += exp_val;
        }
        uniform float sum1 = reduce_add(sum);
        foreach (j = 0 ... N) {
            if (j < N) {
                float val = QK_t[i * N + j];
                val /= sum1;
                QK_t[i * N + j] = val;
            }   
        }
    }
    // softmax(QK^t) * V
    for (uniform int i = 0; i < N; i++) {
        foreach(j = 0 ... d){
            float sum = 0;
            for(uniform int k = 0; k < N; k++) {
                float val1 = QK_t[i * N + k];
                float val2 = V[k * d + j];
                sum += val1 * val2;
            }
            uniform float sum1 = reduce_add(sum);
            O[i * d + j] = sum;
        }
    }
}

// part2

export void part2_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform float QK_t[],
    uniform float V[],
    uniform float O[],
    uniform const int N,
    uniform const int d,
    uniform const int L
){
    // QK^t
    for (uniform int i = 0; i < N; i += L) {

        for (uniform int j = 0; j < N; j += L) {
            
            for (uniform int k = 0; k < d; k += L) {

                for (uniform int i1 = i; i1 < min(i + L, N); i1++) {
                    
                    for (uniform int j1 = j; j1 < min(j + L, N); j1++) {
                        float sum = 0;
                        foreach (k1 = k ... min(k + L, d)) {
                            float val1 = Q[i1 * d + k1];
                            float val2 = K[j1 * d + k1];
                            sum += val1 * val2;
                        }
                        uniform float sum1 = reduce_add(sum);
                        uniform float sum2 = QK_t[i1 * N + j1];
                        if(programIndex == 0) {
                            QK_t[i1 * N + j1] = sum1 + sum2;
                        }
                    }

                }

            }

        }
    }
    // softmax(QK^t)
    for (uniform int i = 0; i < N; i++) {
        float sum = 0;
        foreach (j = 0 ... N) {
            float val = QK_t[i * N + j];
            float exp_val = exp(val);
            QK_t[i * N + j] = exp_val;
            sum += exp_val;
        }
        uniform float sum1 = reduce_add(sum);
        foreach (j = 0 ... N) {
            if (j < N) {
                float val = QK_t[i * N + j];
                val /= sum1;
                QK_t[i * N + j] = val;
            }   
        }
    }

    // softmax(QK^t) * V
    for (uniform int i = 0; i < N; i += L) {

        for (uniform int j = 0; j < d; j += L) {

            for (uniform int k = 0; k < N; k += L) {

                for (uniform int i1 = i; i1 < min(i + L, N); i1++) {
                    
                    foreach (j1 = j ... min(j + L, d)) {
                        float val = 0;
                        for (int k1 = k; k1 < min(k + L, N); k1++) {
                            float val1 = QK_t[i1 * N + k1];
                            float val2 = V[k1 * d + j1];
                            val += val1 * val2;
                        }
                        O[i1 * d + j1] = val + O[i1 * d + j1];
                    }

                }

            }

        }
    }
}

// part3
export void part3_ispc(
    uniform const float Q[],
    uniform const float K[],
    uniform const float V[],
    uniform float O[],
    uniform float ORow[],
    uniform const int N,
    uniform const int d,
    uniform const int i
){
    uniform float exp_sum = 0;
    for (uniform int j = 0; j < N; j++) {
        float val = 0;
        foreach (k = 0 ... d) {
            float val1 = Q[i * d + k];
            float val2 = K[j * d + k];
            val += val1 * val2;
        }
        uniform float sum = reduce_add(val);
        if (programIndex == 0) {
            uniform float exp_val = exp(sum);
            ORow[j] = exp_val;
            exp_sum += exp_val;
        }
    }
    foreach (j = 0 ... d) {
        float val = 0;
        for(int k = 0; k < N; k++) {
            float val1 = ORow[k];
            float val2 = V[k * d + j];
            val += val1 * val2;
        }
        O[i * d + j] = val / exp_sum;
    }
}

// part4
export void part4_ispc(
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
    // load Qi, Oi, li
    int step = b * H * N * d + h * N * d;
    for (int i1 = i; i1 < min(N, i + Br); i1++) {
        int i_local = i1 - i; 
        foreach (k = 0 ... d) {
            Qi[i_local * d + k] = Q[step + i1 * d + k];
            Oi[i_local * d + k] = O[step + i1 * d + k];
        }
        // li[i_local] = l[i1];
    }
    foreach(i1 = i ... min(N, i + Br)) {
        li[i1 - i] = l[i1];
    }
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
        foreach (k = 0 ... d) {
            float val = 0;
            for (int j1 = j; j1 < min(N, j + Bc); j1++) {
                int j_local = j1 - j;
                float val1 = Pij[i_local * Bc + j_local];
                float val2 = Vj[j_local * d + k];
                val += val1 * val2;
            }
            val += Oi[i_local * d + k] * li[i_local]; 
            Oi[i_local * d + k] = val / lnew[i_local];
        }
    }
    // Write Oi, lnew to O and L;
    for (uniform int i1 = i; i1 < min(N, i + Br); i1++) {
        uniform int i_local = i1 - i;
        l[i1] = lnew[i_local];
        foreach (k = 0 ... d) {
            float val = Oi[i_local * d + k];
            O[step + i1 * d + k] = val;
        }
    }
}