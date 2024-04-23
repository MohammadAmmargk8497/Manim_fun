import numpy as np
from manim import *

class AdditiveAttention(Scene):
    def construct(self):
        # Define matrices and vectors
        W1 = np.random.randint(0, 10, size=(8, 5))
        q = np.random.randint(0, 10, size=(5, 1))
        W2 = np.random.randint(0, 10, size=(8, 5))
        k = np.random.randint(0, 10, size=(5, 1))
        b = np.random.randint(0, 10, size=(8, 1))
        w = np.random.randint(0, 10, size=(8, 1))

        # Compute intermediate results
        W1_q = np.dot(W1, q)
        W2_k = np.dot(W2, k)

        # Add bias and the matrices
        r = W1_q + W2_k + b
        
        
        Norm_R =  r/np.linalg.norm(r)
        Act_R = 1/(1 + np.exp(-Norm_R))
        fin = np.dot(np.transpose(Act_R),w)

        # Draw additive attention equation
        equation_text = MathTex("Score(q;k_l)", "=", "w^T", ".", "\sigma", "(", "W_1 \\times q", "+", "W_2 \\times k", "+", "b", ")").scale(1)
        equation_text.set_color(WHITE)
        equation_text.move_to(3 * UP)
        equation_box = SurroundingRectangle(equation_text, buff=0.1)
        W1_text = MathTex("W_1").scale(0.8)
        W1_text.set_color_by_gradient(BLUE)
        W2_text = MathTex("W_2").scale(0.8)
        W2_text.set_color_by_gradient(BLUE)
        q_text =MathTex("q").scale(0.8)
        q_text.set_color_by_gradient(GREEN)
        k_text = MathTex("k_l").scale(0.8)
        k_text.set_color_by_gradient(GREEN)
      
        Lin_box = SurroundingRectangle(equation_text[6:11], buff=0.1)
        box_W1_q = SurroundingRectangle(equation_text[6], buff=0.1)
        box_W2_k = SurroundingRectangle(equation_text[8], buff=0.1)
        box_b = SurroundingRectangle(equation_text[10], buff=0.1)
        box_activation = SurroundingRectangle(equation_text[4], buff=0.1)
        w_trans = SurroundingRectangle(equation_text[2], buff=0.1)
        

       
        

        # Draw matrices and vectors
        matrix_W1 = self.draw_matrix(W1, color=BLUE)
        matrix_W1 = VGroup(matrix_W1, W1_text)
        W1_text.next_to(matrix_W1, DOWN)
        matrix_W1.to_edge(LEFT)


        matrix_q = self.draw_matrix(q, color=GREEN)
        matrix_q = VGroup(matrix_q, q_text)
        q_text.next_to(matrix_q, DOWN)
        matrix_q.to_edge(RIGHT)
        matrix_W1_q = self.draw_matrix(W1_q)
        matrix_W2_k = self.draw_matrix(W2_k)
        matrix_W2 = self.draw_matrix(W2, color=BLUE)
        matrix_W2 = VGroup(matrix_W2, W2_text)
        W2_text.next_to(matrix_W2, DOWN)
        matrix_k = self.draw_matrix(k, color=GREEN)
        matrix_k = VGroup(matrix_k, k_text)
        k_text.next_to(matrix_k, DOWN)
        matrix_k.to_edge(RIGHT)
        vector_b = self.draw_matrix(b, color=RED)
        matrix_r = self.draw_matrix(r)
        matrix_r = VGroup(matrix_r, q_text)
        Norm_R = self.draw_matrix(Norm_R)
        Act_R = self.draw_matrix(Act_R)
        score_vector = self.draw_matrix(np.transpose(w))
        score_vector.to_edge(LEFT)
        score = self.draw_matrix(fin)
        result_text = Text("Result Matrix", font_size=24, color=WHITE).to_edge(UP)
       

        # Animate each step
        self.play(Write(equation_text))
        self.wait(1)
        self.play(ShowCreationThenFadeOut(equation_box), run_time = 3)
        self.wait(1)
        
        self.play(Write(matrix_W1), Write(matrix_q))
        self.wait(1)
        self.play(ShowCreationThenFadeOut(box_W1_q))
        self.play(ReplacementTransform(matrix_W1, matrix_W1_q), ReplacementTransform(matrix_q, matrix_W1_q))
        #self.play(ReplacementTransform(W1_q, W1_q))
        self.wait(1)
        self.play(matrix_W1_q.animate.shift(LEFT*5))
        self.wait(1)
        self.play(Write(matrix_W2), Write(matrix_k))
        self.wait(1)
        self.play(ShowCreationThenFadeOut(box_W2_k))
        self.play(ReplacementTransform(matrix_W2, matrix_W2_k), ReplacementTransform(matrix_k, matrix_W2_k ))
        #self.play(ReplacementTransform(W2_k, W2_k))
        self.wait(1)
        self.play(matrix_W2_k.animate.shift(LEFT*4))
        self.wait(1)
        self.play(ShowCreationThenFadeOut(box_b))
        self.play(Write(vector_b))
        
        self.play(ShowCreationThenFadeOut(Lin_box), run_time = 3)
        self.wait(1)
        
        self.play(ReplacementTransform(matrix_W1_q, matrix_r), ReplacementTransform(matrix_W2_k, matrix_r), ReplacementTransform(vector_b, matrix_r))
        self.remove(matrix_W1_q)
        self.remove(matrix_W2_k)
        self.wait(1)
        
        self.play(ReplacementTransform(matrix_r, Norm_R))
        self.remove(matrix_r)
       
        self.remove(vector_b)
       

        
        

        self.wait(0.5)


       
        
        

        self.wait(1)
        self.play(ShowCreationThenFadeOut(box_activation), run_time = 3)
        self.play(ReplacementTransform(Norm_R, Act_R))
        self.wait(1)
        self.play(Write(score_vector))
        self.play(ShowCreationThenFadeOut(w_trans), run_time = 3)
        self.wait(0.5)
        self.play(ReplacementTransform(Act_R, score), ReplacementTransform(score_vector, score))
        self.wait(0.5)
        self.play(Flash(score, flash_radius=1, num_lines=20))


       
    


    def draw_matrix(self, matrix, color = WHITE):
            grid = VGroup()
            rows, cols = matrix.shape
            cell_size = 0.5
            for i in range(rows):
                for j in range(cols):
                    # Create square with fill_opacity=0 to make it transparent
                    square = Square(side_length=cell_size, fill_opacity=0, stroke_width=1)
                    # Set the position of the square
                    square.move_to((i - (rows - 1) / 2) * cell_size * UP + (j - (cols - 1) / 2) * cell_size * RIGHT)
                    grid.add(square)
                    # Add number to the square
                    num = Text(f"{matrix[i][j]:.2f}", font_size=13, color = color)
                    num.move_to(square.get_center())
                    grid.add(num)
            return grid


    

    def draw_vector(self, vector, color=WHITE):
        vec = VGroup()
        length = len(vector)
        cell_size = 0.5
        for i in range(length):
            square = Square(side_length=cell_size, fill_opacity=0, stroke_width=1)
            num = Text(f"{vector[i][0]:.2f}", font_size=13, color=color)
            num.move_to(square.get_center())
            vec.add(square)
            vec.add(num)
        return vec


# Render the animation
if __name__ == "__main__":
    AdditiveAttention().render()
