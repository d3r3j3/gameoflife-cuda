#include <imgui.h>
#include <stdio.h>

#include <iostream>
#include <vector>
#include "windows.hpp"

// global counter for the number of iterations
int counter = 0;
int* start_state;
int* cpu_cells;
int* gpu_cells;
int* cudnn_cells;
int* shmem_cells;

static State state = createState();

void menuWindow(bool* p_open, State* state, Grid* grid, int width, int height){
    if (!*p_open) {
        return;
    }

    // create menu window fixed to the left and can't be closed or moved
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(200, height), ImGuiCond_FirstUseEver);
    ImGui::Begin("Menu", p_open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    // Show frames per second
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

    // iter range from 1 to 10000
    ImGui::SliderInt("Iters", &grid->max_iter, 1, 1000);

    // show the grid size
    ImGui::Text("Set Grid and Cell Size");
    ImGui::Text("Grid Size: %d x %d", grid->getWidth(), grid->getHeight());

    // update the grid size should square
    static int gwidth = grid->getWidth();
    static int gheight = grid->getHeight();
    static int cell_size = 10;
    ImGui::SliderInt("NxN", &gwidth, 1, height / cell_size);
    ImGui::SliderInt("Cell Size", &cell_size, 1, 50);

    if (ImGui::Button("Update Grid")) {
        grid->clear();
        grid->resize(gwidth, gwidth, cell_size);
    }

    // run init
    if (ImGui::Button("Init Grid")) {
        negateState(state, INIT);
        if (getState(state, INIT)) {
            setState(state, RUN, false);
        }
        cout << "Init\n";
    }

    // if init is true, show dropdown for init
    if (getState(state, INIT)) {
        // grid randomize button
        if (ImGui::Button("Randomize")) {
            grid->randomize();
        }

        // randomize with seed
        static int seed = 0;
        static float density = 0.5;
        if (ImGui::Button("Randomize with Seed")) {
            grid->randomize(seed, density);
        }
        ImGui::InputInt("Seed", &seed);
        ImGui::InputFloat("Density", &density);

        // allow user to select the initial state by clicking on the cells
        ImGui::Text("Set Initial State");
        static bool set_state = false;
        if (ImGui::Button("Set State")) {
            set_state = !set_state;
        }
        ImGui::SameLine();
        ImGui::Text("Set State: %s", set_state ? "True" : "False");

        // if set state is true, allow user to click on the cells
        if (set_state) {
            ImVec2 mouse_pos = ImGui::GetMousePos();
            int x = (int)mouse_pos.x - 200;
            int y = (int)mouse_pos.y;
            if (x >= 0 && x < grid->getWidth() * grid->cell_size && y >= 0 && y < grid->getHeight() * grid->cell_size) {
                int row = y / grid->cell_size;
                int col = x / grid->cell_size;
                ImGui::GetWindowDrawList()->AddRect(ImVec2(col * grid->cell_size + 200, row * grid->cell_size), ImVec2((col + 1) * grid->cell_size + 200, (row + 1) * grid->cell_size), IM_COL32(255, 255, 255, 255));
                if (ImGui::IsMouseClicked(0)) {
                    grid->setCellAlive(row, col, !grid->isCellAlive(row, col));
                }
            }
        }

        // clear grid
        if (ImGui::Button("Clear")) {
            grid->clear();
        }
    }

    // run cpu
    if (ImGui::Button("CPU")) {
        if (!getState(state, CPU)) {
            grid->cpu_time = 0.0f;
        }

        negateState(state, CPU);
        cout << "CPU STATE SWITCHED\n";
    }
    ImGui::SameLine();
    ImGui::Text("CPU State: %s", getState(state, CPU) ? "True" : "False");

    // run gpu
    if (ImGui::Button("GPU")) {
        if (!getState(state, GPU)) {
            grid->gpu_time = 0.0f;
        }
        negateState(state, GPU);

        cout << "GPU STATE SWITCHED\n";
    }
    ImGui::SameLine();
    ImGui::Text("GPU State: %s", getState(state, GPU) ? "True" : "False");

    // run cudnn
    if (ImGui::Button("CUDNN")) {
        if (!getState(state, CUDNN)) {
            grid->cudnn_time = 0.0f;
        }
        negateState(state, CUDNN);

        cout << "CUDNN STATE SWITCHED\n";
    }
    ImGui::SameLine();
    ImGui::Text("CUDNN State: %s", getState(state, CUDNN) ? "True" : "False");

    // run
    if (ImGui::Button("Run")) {
        negateState(state, RUN);
        if (getState(state, RUN)) {
            grid->cpu_time = 0.0f;
            grid->gpu_time = 0.0f;
            grid->cudnn_time = 0.0f;
            setState(state, INIT, false);
            start_state = new int[grid->getWidth() * grid->getHeight()];
            memcpy(start_state, grid->cells, grid->getWidth() * grid->getHeight() * sizeof(int));
            counter = 0;

            cout << "Running\n";
        } else {
            cout << "Stopped\n";
        }
    }

    // show the time taken by cpu and gpu
    ImGui::Text("CPU Time  : %.3f ms", grid->cpu_time);
    ImGui::Text("GPU Time  : %.3f ms", grid->gpu_time);
    ImGui::Text("CUDNN Time: %.3f ms", grid->cudnn_time);

    ImGui::End();
}

// main window
void mainWindow(bool* p_open, State* state, Grid* grid, int width, int height) {
    if (!*p_open) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(200, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(width - 200, height), ImGuiCond_FirstUseEver);
    ImGui::Begin("Main", p_open,  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    // draw grid outline
    ImGui::GetWindowDrawList()->AddRect(ImVec2(200, 0), ImVec2(grid->getWidth() * grid->cell_size + 200, grid->getHeight() * grid->cell_size), IM_COL32(255, 255, 255, 255));

    if (grid->cell_size >= 5) {
        // draw grid lines
        for (int i = 0; i < grid->getWidth(); i++) {
            // shift 200 in x to make the cells visible
            ImGui::GetWindowDrawList()->AddLine(ImVec2(i * grid->cell_size + 200, 0), ImVec2(i * grid->cell_size + 200, grid->getHeight() * grid->cell_size), IM_COL32(255, 255, 255, 255));
        }

        for (int i = 0; i < grid->getHeight(); i++) {
            // shift by 200 to make the cells visible
            ImGui::GetWindowDrawList()->AddLine(ImVec2(200, i * grid->cell_size), ImVec2(grid->getWidth() * grid->cell_size + 200, i * grid->cell_size), IM_COL32(255, 255, 255, 255));
        }
    }

    // draw the cells
    grid->draw();

    // run the simulation
    if (getState(state, RUN)) {
        // run cpu first
        // if counter is less than max_iter
        // run gpu if counter is between max_iter and 2 * max_iter

        if (counter < grid->max_iter) {
            if (getState(state, CPU)) {
                grid->cpu_time += grid->update();
                counter++;
            } else {
                counter = grid->max_iter;
            }

            // save final state
            if (counter == grid->max_iter) {
                cpu_cells = new int[grid->getWidth() * grid->getHeight()];
                memcpy(cpu_cells, grid->cells, grid->getWidth() * grid->getHeight() * sizeof(int));
                grid->setState(start_state);
            }
        } else if (counter < 2 * grid->max_iter) {
            if (getState(state, GPU)) {
                grid->gpu_time += grid->updateGPU();
                counter++;
            } else {
                counter = 2 * grid->max_iter;
            }

            // save final state
            if (counter == 2 * grid->max_iter) {
                gpu_cells = new int[grid->getWidth() * grid->getHeight()];
                memcpy(gpu_cells, grid->cells, grid->getWidth() * grid->getHeight() * sizeof(int));
                grid->setState(start_state);
            }
        } else if (counter < 3 * grid->max_iter) {
            if (getState(state, CUDNN)) {
                grid->cudnn_time += grid->updateCUDNN();
                counter++;
            } else {
                counter = 3 * grid->max_iter;
            }

            // save final state
            if (counter == 3 * grid->max_iter) {
                cudnn_cells = new int[grid->getWidth() * grid->getHeight()];
                memcpy(cudnn_cells, grid->cells, grid->getWidth() * grid->getHeight() * sizeof(int));
                grid->setState(start_state);
            }
        } else {
            // compare the final states
            int cpu_diff = 0;
            int gpu_diff = 0;
            int cudnn_diff = 0;

            for (int i = 0; i < grid->getWidth() * grid->getHeight(); i++) {
                if (cpu_cells[i] != gpu_cells[i]) {
                    cpu_diff++;
                }

                if (gpu_cells[i] != cudnn_cells[i]) {
                    gpu_diff++;
                }

                if (cpu_cells[i] != cudnn_cells[i]) {
                    cudnn_diff++;
                }

            }

            cout << "Differences\n";
            cout << "CPU vs GPU: " << cpu_diff << endl;
            cout << "GPU vs CUDNN: " << gpu_diff << endl;
            cout << "CPU vs CUDNN: " << cudnn_diff << endl;
            setState(state, RUN, false);
            counter = 0;
        }
    }

    ImGui::End();
}

// draw the windows
void drawWindows(bool* m_open, bool* main_open, Grid* grid, int width, int height) {
    menuWindow(m_open, &state, grid, width, height);
    mainWindow(main_open, &state, grid, width, height);
}

void negateState(State* state, int bit) {
    void* ptr = state;
    int* int_ptr = (int*)ptr;
    *int_ptr ^= 1 << bit;
};

void negateAll(State* state) {
    void* ptr = state;
    int* int_ptr = (int*)ptr;
    *int_ptr = ~*int_ptr;
};

void setState(State* state, int bit, bool value) {
    void* ptr = state;
    int* int_ptr = (int*)ptr;
    if (value) {
        *int_ptr |= 1 << bit;
    } else {
        *int_ptr &= ~(1 << bit);
    }
};

bool getState(State* state, int bit) {
    void* ptr = state;
    int* int_ptr = (int*)ptr;
    return *int_ptr & (1 << bit);
};

State createState() {
    State state = {false, false, false, false, false};
    return state;
};