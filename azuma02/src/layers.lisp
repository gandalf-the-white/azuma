(in-package :azuma)

(defun mat-transpose (matrix)
  (let* ((rows (length matrix))
         (cols (length (aref matrix 0)))
         (result (make-array (list cols))))
    (dotimes (j cols)
      (let ((row (make-array rows)))
        (dotimes (i rows)
          (setf (aref row i)(aref (aref matrix i) j)))
        (setf (aref result j) row)))
    result))

(defclass dense-layer ()
  ((weights :initarg :weights
            :accessor layer-weights
            :documentation "Weight matrix (vector of vectors).")
   (bias :initarg :bias
         :accessor layer-bias)
   (activation :initarg :activation
               :accessor layer-activation
               :documentation "Activation function or NIL for linear.")
   (last-input :accessor layer-last-input
               :initform nil
               :documentation "Layer input")
   (last-z :accessor layer-last-z
           :initform nil
           :documentation "Linear output before activation")
   (last-output :accessor layer-last-output
                :initform nil
                :documentation "Output after activation")
   (grad-weights :accessor layer-grad-weights
                 :initform nil
                 :documentation "Accumulated weight gradient")
   (grad-bias :accessor layer-grad-bias
              :initform nil
              :documentation "Accumulated bias gradient")
   (accum-grad-weights :accessor layer-accum-grad-weights
                       :initform nil
                       :documentation "Weight gradient accumulation")
   (accum-grad-bias :accessor layer-accum-grad-bias
                    :initform nil
                    :documentation "Bias gradient accumulation")))

(defgeneric forward-layer (layer input))

(defmethod forward-layer ((layer dense-layer) input)
  (let* ((z (vec-add (mat-vec-mul (layer-weights layer) input)
                     (layer-bias layer)))
         (act (layer-activation layer))
         (out (if act (funcall act z) z)))
    ;; Backward step storage
    (setf (layer-last-input layer)  input
          (layer-last-z layer) z
          (layer-last-output layer) out)
    out))

(defgeneric backward-layer (layer dout))

(defmethod backward-layer ((layer dense-layer) dout)
  (let* ((W (layer-weights layer))
         (x (layer-last-input layer))
         (z (layer-last-z layer))
         (act (layer-activation layer))
         (dz (cond
               ;; ReLU case
               ((eq act #'relu-vector)
                (vec-hadamard dout (relu-deriv-vec z)))
               ;; no activation
               (t
                dout)))
         ;; Gradient relative to the current stack 
         (grad-W (outer-product dz x))
         (grad-b dz)
         ;; Gradient send back to the previous stack
         (WT (mat-transpose W))
         (din (mat-vec-mul WT dz)))
    (setf (layer-grad-weights layer) grad-W
          (layer-grad-bias layer) grad-b)
    din))

(defun apply-gradients! (layer learning-rate)
  (let ((W (layer-weights layer))
        (b (layer-bias layer))
        (gW (layer-grad-weights layer))
        (gB (layer-grad-bias layer)))
    ;; W := W - lr * dW
    (dotimes (i (length W))
      (dotimes (j (length (aref W i)))
        (setf (aref (aref W i) j)
              (- (aref (aref W i) j)
                 (* learning-rate
                    (aref (aref gW i) j))))))
    ;; b := b - lr * gB
    (dotimes (i (length b))
      (setf (aref b i)
            (- (aref b i)
               (* learning-rate (aref gB i)))))
    layer))
