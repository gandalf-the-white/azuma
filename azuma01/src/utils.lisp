(in-package :azuma)

(defun dot-product (v1 v2)
  (reduce #'+ (map 'list #'* v1 v2)))

;; return scalar
;; Example
;; v1 = #(1 2)
;; v2 = #(3 4)
;; 1 + 2 + 3 + 4 = 10

(defun mat-vec-mul (matrix vec)
  "Multiplie chaque ligne de matrix par vec."
  (map 'vector
       (lambda (row)
         (dot-product row vec))
       matrix))

;; return vector
;; Example:
;; |1 2| * |1| = |1*2 + 2*2| = | 5|
;; |3 4|   |2|   |3*1 + 4*2|   |11|

(defun vec-add (v1 v2)
  (map 'vector #'+ v1 v2))

;; return vector
;; Example:
;; v1 = '(1 2)
;; v2 = '(3 4)
;; |1 + 3| = | 4|
;; |2 + 4|   | 6|

(defun relu (x)
  (max 0 x))

;; return max of numbers

(defun relu-vec (v)
  (map 'vector #'relu v))

;; return relu result as a vector
;; (relu-vec '(4 7 2))
;; #(4 7 2)

(defun relu-deriv-vec (z)
  "Renvoie un vecteur de 0/1 selon z>0"
  (map 'vector (lambda (x) (if (> x 0) 1 0)) z))

(defun vec-sub (v1 v2)
  (map 'vector #'- v1 v2))

(defun vec-hadamard (v1 v2)
  "Produit element par element"
  (map 'vector #'* v1 v2))

;; exemple
;; (vec-hadamard #(1 3) #(2 6))
;; #(2 18)

(defun outer-product (v1 v2)
  "Retourne une matrice = v1 (colonne) * v2 (ligne)"
  (let ((n1 (length v1))
        (n2 (length v2)))
    (let ((m (make-array (list n1 n2))))
      (dotimes (i n1)
        (dotimes (j n2)
          (setf (aref m i j)
                (* (aref v1 i)(aref v2 j)))))
      ;; On renvoie un vecteur de vecteurs pour rester coherent
      (map 'vector
           (lambda (i)
             (let ((row (make-array n2)))
               (dotimes (j n2)
                 (setf (aref row j) (aref m i j)))
               row))
           (loop for i below n1 collect i)))))

(defun mat-transpose (matrix)
  "Retourne la transposee d'une matrice"
  (let* ((rows (length matrix))
         (cols (length (aref matrix 0)))
         (result (make-array (list cols))))
    (dotimes (j cols)
      (let ((row (make-array rows)))
        (dotimes (i rows)
          (setf (aref row i) (aref (aref matrix i) j)))
        (setf (aref result j) row)))
    result))

(defun forward-mlp (x)
  ;; Couche cachee
  (let* ((z1 (vec-add (mat-vec-mul *w1* x) *b1*))
         (h (relu-vec z1))
         (z2 (vec-add (mat-vec-mul *w2* h) *b2*)))
    z2))

(defun make-like-matrix (matrix &key (initial-element 0.0))
  (let* ((rows (length matrix))
         (cols (length (aref matrix 0)))
         (m (make-array rows)))
    (dotimes (i rows)
      (let ((row (make-array cols)))
        (dotimes (j cols)
          (setf (aref row j) initial-element))
        (setf (aref m i) row)))
    m))

(defun make-like-vector (vec &key (initial-element 0.0))
  (let* ((n (length vec))
         (v (make-array n)))
    (dotimes (i n)
      (setf (aref v i) initial-element))
    v))

(defun mat-add! (a b)
  "A := A + B (in-place)."
  (dotimes (i (length a))
    (dotimes (j (length (aref a i)))
      (incf (aref (aref a i) j)
            (aref (aref b i) j))))
  a)

(defun vec-add! (a b)
  "A := A + B (in-place)."
  (dotimes (i (length a))
    (incf (aref a i) (aref b i)))
  a)

(defun mat-scale! (a factor)
  "A := factor * A (in-place)."
  (dotimes (i (length a))
    (dotimes (j (length (aref a i)))
      (setf (aref (aref a i) j)
            (* factor (aref (aref a i) j)))))
  a)

(defun vec-scale! (a factor)
  "A := factor * A (in-place)."
  (dotimes (i (length a))
    (setf (aref a i) (* factor (aref a i))))
  a)

(defun make-index-vector (n)
  (let ((v (make-array n)))
    (dotimes (i n)
      (setf (aref v i) i))
    v))

(defun shuffle-vector! (v)
  "Shuffle in-place (Fisher–Yates)."
  (let ((n (length v)))
    (dotimes (i n)
      (let* ((j (+ i (random (- n i))))
             (tmp (aref v i)))
        (setf (aref v i) (aref v j)
              (aref v j) tmp))))
  v)
