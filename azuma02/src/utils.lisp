(in-package :azuma)

(defun dot-product (v1 v2)
  (reduce #'+ (map 'list #'* v1 v2)))

(defun mat-vec-mul (matrix vec)
  "Multiply each row of MATRIX by the vector VEC.
 MATRIX is a vector of vectors, VEC is a vector."
  (map 'vector
       (lambda (row)
         (reduce #'+ (map 'list #'* row vec)))
       matrix))

(defun vec-add (v1 v2)
  (map 'vector #'+ v1 v2))

(defun vec-sub (v1 v2)
  (map 'vector #'- v1 v2))

(defun relu (x)
  (max 0 x))

(defun relu-vector (v)
  (map 'vector #'relu v))

(defun relu-deriv-vec (z)
  "Send back a vector 0/1 according to z"
  (map 'vector (lambda (x) (if (> x 0) 1 0)) z))

(defun vec-hadamard (v1 v2)
  "Multiply element per element"
  (map 'vector #'* v1 v2))

(defun outer-product (v1 v2)
  "Buit a matrix of  v1 columns and v2 rows"
  (let ((n1 (length v1))
        (n2 (length v2)))
    (let ((m (make-array (list n1 n2))))
      (dotimes (i n1)
        (dotimes (j n2)
          (setf (aref m i j)
                (* (aref v1 i)(aref v2 j)))))
      ;; Send back vector of vectors
      (map 'vector
           (lambda (i)
             (let ((row (make-array n2)))
               (dotimes (j n2)
                 (setf (aref row j)(aref m i j)))
               row))
           (loop for i below n1 collect i)))))

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

