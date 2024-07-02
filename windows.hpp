#include "objects.hpp"

enum GameStates {
    CPU = 0,
    GPU = 1,
    CUDNN = 2,
    INIT = 3,
    RUN = 4
};

typedef struct {
    bool cpu_state;
    bool gpu_state;
    bool cudnn_state;
    bool init_state;
    bool run_state;
} State;

void negateState(State* state, int bit);

void negateAll(State* state);

void setState(State* state, int bit, bool value);

bool getState(State* state, int bit);

State createState();

void menuWindow(bool* p_open, State *state, Grid* grid, int width, int height);
void mainWindow(bool* p_open, State *state, Grid* grid, int width, int height);
void drawWindows(bool* m_open, bool* main_open, Grid* grid, int width, int height);