#include <chrono>

void wtime(double* t) {
    static int sec = -1;
    auto now = std::chrono::high_resolution_clock::now();
    if (sec < 0)
        sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    *t = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() * 1.0e-6;
}

static double start[16], elapsed[16];

double elapsed_time() {
    double t;
    wtime(&t);
    return t;
}

void timer_clear(int n) {
    elapsed[n] = 0.0;
}

void timer_start(int n) {
    start[n] = elapsed_time();
}

void timer_stop(int n) {
    double t, now;
    now = elapsed_time();
    t = now - start[n];
    elapsed[n] += t;
}

double timer_read(int n) {
    return elapsed[n];
}