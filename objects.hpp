
using namespace std;
#include <vector>

class Grid {
    public:
        int* cells;
        int width;
        int height;
        int cell_size = 10;
        float cpu_time;
        float gpu_time;
        float cudnn_time;
        int max_iter = 1000;
        Grid(int width, int height);
        Grid(int width, int height, int* state);
        int  getIdx(int row, int col);
        float update();
        float* cudnnCalcNeighbors();
        float updateCUDNN();
        void compareCUDNN(float* cudnn_output, int* cpu_output);
        void calcNeighbors(int* out);
        void gpuCalcNeighbors(int* out);
        void compareNeighbors(int* cpu, int* gpu);
        float updateGPU();
        void draw();
        void drawCell(int row, int col, int size);
        void clear();
        void randomize();
        void randomize(int seed, float density);
        void setState(int* state);
        void setCellAlive(int row, int col, bool alive);
        bool isCellAlive(int row, int col);
        int getWidth();
        int getHeight();
        void resize(int width, int height, int size);
};

