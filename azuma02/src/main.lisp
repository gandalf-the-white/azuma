(defpackage :azuma
  (:use #:cl))


(in-package :azuma)

;;==========================================
;; T O O L 
;;==========================================

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

;;==========================================
;; C L O S 
;;==========================================

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

(defclass mlp ()
  ((layers :initarg :layers
           :accessor mlp-layers
           :documentation "List of layers (DENSE-LAYER instances).")))

;;==========================================
;; F O R W A R D
;;==========================================

(defgeneric forward (network input))

(defmethod forward ((network mlp) input)
  (reduce (lambda (x layer)
            (forward-layer layer x))
          (mlp-layers network)
          :initial-value input))

(defun forward-mlp (x &optional (W1 *W1*) (b1 *b1*) (W2 *W2*) (b2 *b2*))
  "Step forward for MLP with only one layer"
  (let* ((z1 (vec-add (mat-vec-mul W1 x) b1))
         (h (relu-vector z1))
         (z2 (vec-add (mat-vec-mul W2 h) b2)))
    ;; z2 -> prediction [CPU_next MEM_next]
    z2))

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

;;==========================================
;; B A C K W A R D 
;;==========================================

(defgeneric backward (network input))

(defmethod backward ((network mlp) dloss-dy)
  (let ((grad dloss-dy))
    (dolist (layer (reverse (mlp-layers network)) grad)
      (setf grad (backward-layer layer grad)))))

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

;;==========================================
;; M S E
;;==========================================

(defun mse-loss-and-grad (y-hat y)
  "Send back 2 values: (loss dL/dy-hat)"
  (let* ((diff (vec-sub y-hat y))     ;; diff = y-hat - y
         (squared (vec-hadamard diff diff))
         (loss (* 0.5 (reduce #'+ squared))))
    (values loss diff)))              ;; dL/dy-hat = y-hat - y 

;;==========================================
;; U P D A T E   G R A D I E N T 
;;==========================================

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

(defun apply-gradients-network! (network learning-rate)
  (dolist (layer (mlp-layers network))
    (apply-gradients! layer learning-rate)))

(defun accumulate-current-gradients! (network)
  (dolist (layer (mlp-layers network))
    (let ((gW (layer-grad-weights layer))
          (gB (layer-grad-bias layer))
          (aW (layer-accum-grad-weights layer))
          (aB (layer-accum-grad-bias layer)))
      ;; aW += gW ; aB += gB
      (mat-add! aW gW)
      (vec-add! aB gB))))

(defun average-accum-gradients-network! (network batch-size)
  (let ((factor (/ 1.0 batch-size)))
    (dolist (layer (mlp-layers network))
      (let ((aW (layer-accum-grad-weights layer))
            (aB (layer-accum-grad-bias layer)))
        (mat-scale! aW factor)
        (vec-scale! aB factor)
        ;; on met ces moyennes dans grad-weights / grad-bias
        (setf (layer-grad-weights layer) aW
              (layer-grad-bias layer)    aB)))))

;;==========================================
;; T R A I N I N G 
;;==========================================

(defun train-one-step (net x y learning-rate)
  (let* ((y-hat (forward net x))
         (loss nil)
         (dloss-dy nil))
    (multiple-value-setq (loss dloss-dy)
      (mse-loss-and-grad y-hat y))
    (format t "Loss before update: ~A~%" loss)
    (backward net dloss-dy)
    (apply-gradients-network! net learning-rate)
    (let* ((y-hat-new (forward net x))
           (loss-new (multiple-value-bind (l _)
                         (mse-loss-and-grad y-hat-new y)
                       (declare (ignore _))
                       l)))
      (format t "Loss after update: ~A~%" loss-new))))

(defun train-one-batch (net xs ys learning-rate)
  (let* ((batch-size (length xs))
         (total-loss 0.0))
    ;; 1) accumulator initialization
    (zero-accum-gradients-network! net)

    ;; 2) loop on examples
    (dotimes (n batch-size)
      (let* ((x (elt xs n))
             (y (elt ys n))
             (y-hat (forward net x))
             (loss nil)
             (dloss-dy nil))
        (multiple-value-setq (loss dloss-dy)
          (mse-loss-and-grad y-hat y))
        (incf total-loss loss)
        ;; backward for this example
        (backward net dloss-dy)
        ;; gradients accumulation for this example
        (accumulate-current-gradients! net)))

    ;; 3) gradients average for this batch
    (average-accum-gradients-network! net batch-size)

    ;; 4) weights update with average gradients
    (apply-gradients-network! net learning-rate)

    ;; 5) send back batch average loss
    (/ total-loss batch-size)))

(defun train-mini-batch (net xs ys index-vector start batch-size learning-rate)
  "Trian on mini-batch defined by INDEX-VECTOR[start .. start+batch-size[."
  (let ((total-loss 0.0))
    ;; 1) accumulator initialization
    (zero-accum-gradients-network! net)

    ;; 2) loop on examples
    (dotimes (k batch-size)
      (let* ((idx (aref index-vector (+ start k)))
             (x   (aref xs idx))
             (y   (aref ys idx))
             (y-hat (forward net x))
             (loss nil)
             (dloss-dy nil))
        (multiple-value-setq (loss dloss-dy)
          (mse-loss-and-grad y-hat y))
        (incf total-loss loss)
        ;; backward + accumulation
        (backward net dloss-dy)
        (accumulate-current-gradients! net)))

    ;; 3) gradients average for this batch
    (average-accum-gradients-network! net batch-size)

    ;; 4) Weight update
    (apply-gradients-network! net learning-rate)

    ;; 5) send back batch average loss
    (/ total-loss batch-size)))

;; (defun train-epochs (net xs ys
;;                      &key
;;                        (epochs 10)
;;                        (batch-size 2)
;;                        (learning-rate 0.01)
;;                        (verbose t))
;;   (let* ((n-samples (length xs)))
;;     (assert (= n-samples (length ys)))
;;     (let ((indices (make-index-vector n-samples)))
;;       (dotimes (epoch epochs net)
;;         ;; shuffle des indices à chaque epoch
;;         (shuffle-vector! indices)
;;         (let ((epoch-loss 0.0)
;;               (n-batches 0))
;;           ;; boucle sur les mini-batches
;;           (loop for start from 0 below n-samples by batch-size do
;;             (let* ((remaining (- n-samples start))
;;                    (current-batch-size (min batch-size remaining))
;;                    (batch-loss (train-mini-batch
;;                                 net xs ys indices
;;                                 start current-batch-size
;;                                 learning-rate)))
;;               (incf epoch-loss batch-loss)
;;               (incf n-batches)))
;;           (let ((avg-loss (/ epoch-loss (max 1 n-batches))))
;;             (when verbose
;;               (format t "Epoch ~D, Average Loss: ~A~%" epoch avg-loss))))))))

(defun train-epochs (net xs ys
                     &key
                       (epochs 10)
                       (batch-size 2)
                       (learning-rate 0.01)
                       (verbose t))
  "Send back a list of (epoch avg-loss)"
  (let* ((n-samples (length xs)))
    (assert (= n-samples (length ys)))
    (let ((indices (make-index-vector n-samples))
          (loss-history '()))  ;; list of  (epoch avg-loss)
      (dotimes (epoch epochs loss-history)
        ;; shuffle index each epoch
        (shuffle-vector! indices)
        (let ((epoch-loss 0.0)
              (n-batches 0))
          ;; loop on mini-batches
          (loop for start from 0 below n-samples by batch-size do
            (let* ((remaining (- n-samples start))
                   (current-batch-size (min batch-size remaining))
                   (batch-loss (train-mini-batch
                                net xs ys indices
                                start current-batch-size
                                learning-rate)))
              (incf epoch-loss batch-loss)
              (incf n-batches)))
          (let ((avg-loss (/ epoch-loss (max 1 n-batches))))
            (when verbose
              (format t "Epoch ~D, average loss moyenne: ~A~%" epoch avg-loss))
            ;; Save (epoch avg-loss)
            (push (list epoch avg-loss) loss-history))))
      ;; push and reverse
      (reverse loss-history))))

;;==========================================
;; I N I T I A L I Z A T I O N
;;==========================================

(defun zero-accum-gradients-network! (network)
  (dolist (layer (mlp-layers network))
    (let ((W (layer-weights layer))
          (b (layer-bias layer)))
      (setf (layer-accum-grad-weights layer)
            (make-like-matrix W :initial-element 0.0)
            (layer-accum-grad-bias layer)
            (make-like-vector b :initial-element 0.0)))))

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


;;==========================================
;; E X P O R T
;;==========================================

(defun export-loss-history-to-dat (loss-history filepath)
  "LOSS-HISTORY list of (epoch avg-loss)."
  (with-open-file (out filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (dolist (entry loss-history)
      (destructuring-bind (epoch loss) entry
        (format out "~D ~F~%" epoch loss)))))

;;==========================================
;; E X A M P L E 
;;==========================================

;; (defparameter *x1* #(0.4 0.5 0.6 0.3))
;; (defparameter *y1* #(0.5 0.4))

;; (defparameter *x2* #(0.2 0.1 0.7 0.9))
;; (defparameter *y2* #(0.8 0.2))

;; (defparameter *xs* (vector *x1* *x2*))
;; (defparameter *ys* (vector *y1* *y2*))

(defparameter *xs*
  (vector
   #(0.4 0.5 0.6 0.3)
   #(0.2 0.1 0.7 0.9)
   #(0.1 0.9 0.3 0.4)
   #(0.6 0.2 0.5 0.8)))

(defparameter *ys*
  (vector
   #(0.5 0.4)
   #(0.8 0.2)
   #(0.3 0.7)
   #(0.9 0.1)))

(defparameter *w1* #(#(0.5 -0.2 0.3 0.1)
                     #(-0.3 0.8 0.2 0.4)))

(defparameter *b1* #(0.0 0.1))

(defparameter *w2* #(#(1.0 0.5)
                     #(-0.5 1.0)))

(defparameter *b2* #(0.1 0.2))

(defparameter *hidden-layer*
  (make-instance 'dense-layer
                 :weights *w1*
                 :bias *b1*
                 :activation #'relu-vector))

(defparameter *output-layer*
  (make-instance 'dense-layer
                 :weights *w2*
                 :bias *b2*
                 :activation nil))

(defparameter *net*
  (make-instance 'mlp
                 :layers (list *hidden-layer* *output-layer*)))


;; (let ((loss-before-1 (multiple-value-bind (l _)
;;                          (mse-loss-and-grad (forward *net* *x1*) *y1*)
;;                        (declare (ignore _))
;;                        l))
;;       (loss-before-2 (multiple-value-bind (l _)
;;                          (mse-loss-and-grad (forward *net* *x2*) *y2*)
;;                        (declare (ignore _))
;;                        l)))
;;   (format t "Loss x1 avant batch: ~A~%" loss-before-1)
;;   (format t "Loss x2 avant batch: ~A~%" loss-before-2)

;;   (let ((batch-loss (train-one-batch *net* *xs* *ys* 0.1)))
;;     (format t "Loss moyenne sur batch (avant update): ~A~%" batch-loss))

;;   ;; après update, on recalcule les losses individuellement
;;   (let ((loss-after-1 (multiple-value-bind (l _)
;;                           (mse-loss-and-grad (forward *net* *x1*) *y1*)
;;                         (declare (ignore _))
;;                         l))
;;         (loss-after-2 (multiple-value-bind (l _)
;;                           (mse-loss-and-grad (forward *net* *x2*) *y2*)
;;                         (declare (ignore _))
;;                         l)))
;;     (format t "Loss x1 après batch: ~A~%" loss-after-1)
;;     (format t "Loss x2 après batch: ~A~%" loss-after-2)))

;; (defparameter *loss-history*
;;   (train-epochs *net* *xs* *ys*
;;                 :epochs 50
;;                 :batch-size 16
;;                 :learning-rate 0.05
;;                 :verbose t))

;; (export-loss-history-to-dat *loss-history* "datas/loss_epochs.dat")
