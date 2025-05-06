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
            for (int k = 0; k < d; k += programCount) {
                int kid = k + programIndex;
                if (kid < d) {
                    float val1 = Q[b * H * N * d + h * N * d + i * d + kid];
                    float val2 = K[b * H * N * d + h * N * d + j * d + kid];
                    val += val1 * val2;
                }
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
        for (int j = 0; j < N; j += programCount) {
            int jid = j + programIndex;
            if (jid < N) {
                float val = QK_t[i * N + jid];
                float exp_val = exp(val);
                QK_t[i * N + jid] = exp_val;
                sum += exp_val;
            }
        }
        uniform float sum1 = reduce_add(sum);
        for (int j = 0; j < N; j += programCount) {
            int jid = j + programIndex;
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
            for(int k = 0; k < N; k += programCount) {
                int kid = k + programIndex;
                if (kid < N) {
                    float val1 = QK_t[i * N + kid];
                    float val2 = V[b * H * N * d + h * N * d + kid * d + j];
                    sum += val1 * val2;
                }
            }
            uniform float sum1 = reduce_add(sum);
            if (programIndex == 0) {
                O[b * H * N * d + h * N * d + i * d + j] = sum1;
            }
        }
    }
}