import numpy as np
from manim import *

class MatrixMultiplication(Scene):
    def construct(self):
        # Define random matrices using NumPy arrays
        matrix_A = np.random.randint(0, 256, size=(5, 3))
        matrix_B = np.random.randint(0, 256, size=(3, 1))

        # Perform matrix multiplication using NumPy
        result_matrix = np.dot(matrix_A, matrix_B)
        normalised_matrix = result_matrix / np.linalg.norm(result_matrix)

        # Define grid dimensions and cell size
        rows_A, cols_A = matrix_A.shape
        rows_B, cols_B = matrix_B.shape[0], matrix_B.shape[1]
        cell_size = 1

        # Add title for matrices
        title_A = Text("Matrix A", font_size=24, color=WHITE)
        title_A.to_edge(UP).shift(3 * LEFT)
        title_B = Text("Matrix B", font_size=24, color=WHITE)
        title_B.to_edge(UP).shift(3 * RIGHT)
        title_result = Text("Result Matrix", font_size=24, color=WHITE)
        title_result.to_edge(UP)

        # Draw grids for matrices A, B, and result
        grid_A = self.draw_grid(matrix_A, rows_A, cols_A, cell_size, font_size=14)
        grid_A.to_edge(LEFT)
        grid_B = self.draw_grid(matrix_B, rows_B, cols_B, cell_size, font_size=14)
        grid_B.to_edge(RIGHT)
        grid_result = self.draw_grid(result_matrix, rows_A, cols_B, cell_size, font_size=14)
        grid_result_normalised = self.draw_grid(normalised_matrix, rows_A, cols_B, cell_size, font_size=14)

        # Perform matrix multiplication animation
        self.play(Write(title_A), Write(title_B))
        self.wait(1)
        self.play(Create(grid_A), Create(grid_B))
        self.wait(1)
        self.play(Transform(grid_A, grid_result), Transform(grid_B, grid_result))
        self.wait(1)
        self.play(Write(title_result))
        self.wait(1)
        self.play(ReplacementTransform(grid_result, grid_result_normalised))

    def draw_grid(self, matrix, rows, cols, cell_size, font_size):
        grid = VGroup()
        for i in range(rows):
            for j in range(cols):
                num = Text(f"{matrix[i][j]:.2f}", font_size=font_size)
                num.move_to((i - (rows - 1) / 2) * cell_size * UP + (j - (cols - 1) / 2) * cell_size * RIGHT)
                grid.add(num)
        return grid

# Render the animation
if __name__ == "__main__":
    MatrixMultiplication().render()
