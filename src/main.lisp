(in-package :azuma)

(defun hello ()
  (format t "Hello world!~%"))

(multiple-value-bind (sec min hour day month year)
    (decode-universal-time (get-universal-time))
  (format t "~4,'0D-~2,'0D-~2,'0DT~2,'0D:~2,'0D:~2,'0D"
          year month day hour min sec))

;; (defparameter *x* #(0.4 0.5 0.6 0.3))
(defparameter *W1* #(#(0.5 -0.2 0.3 0.1) #(-0.3 0.8 0.2 0.4)))
(defparameter *b1* #(0.0 0.1))
(defparameter *W2* #(#(1.0 0.5) #(-0.5 1.0)))
(defparameter *b2* #(0.1 0.2))

;;-----------------------------------
;; Outils
;;-----------------------------------

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

;; (defun forward-mlp (x)
;;   ;; Couche cachee
;;   (let* ((z1 (vec-add (mat-vec-mul *w1* x) *b1*))
;;          (h (relu-vec z1))
;;          (z2 (vec-add (mat-vec-mul *w2* h) *b2*)))
;;     z2))

(defclass dense-layer ()
  ((weights
    :initarg :weights
    :accessor layer-weights
    :documentation "Matrice des poids (vecteur de vecteurs)")
   (bias
    :initarg :bias
    :accessor layer-bias
    :documentation "Vecteur de biais")
   (activation
    :initarg :activation
    :accessor layer-activation
    :documentation "Fonction d'activation ou NIL pour lineaire")
   (last-input
    :initarg :last-input
    :accessor layer-last-input
    :initform nil
    :documentation "Entree de la couche")
   (last-z
    :accessor layer-last-z
    :initform nil
    :documentation "Sortie lineaire avant activation")
   (last-output
    :accessor layer-last-output
    :initform nil
    :documentation "Sortie apres activation")
   (grad-weights
    :accessor layer-grad-weights
    :initform nil
    :documentation "Gradients accumulees pour cette passe")
   (grad-bias
    :accessor layer-grad-bias
    :initform nil
    :documentation "Biais accumules pour cette passe")))

(defgeneric forward-layer (layer input))

(defmethod forward-layer ((layer dense-layer) input)
  (let* ((z (vec-add (mat-vec-mul (layer-weights layer) input)
                     (layer-bias layer)))
         (act (layer-activation layer))
         (out (if act
                  (funcall act z))))
    (setf (layer-last-input layer) input
          (layer-last-z layer) z
          (layer-last-output layer) out)
    out))

(defclass mlp ()
  ((layers
    :initarg :layers
    :accessor mlp-layers
    :documentation "Liste de couches (instance de dense-layer)")))

;;----------------------------------
;; Forward 
;;----------------------------------

(defgeneric forward (network input))

(defmethod forward ((network mlp) input)
  (reduce (lambda (x layer)
            (forward-layer layer x))
          (mlp-layers network)
          :initial-value input))

;;----------------------------------
;; Backward 
;;----------------------------------

(defgeneric backward-layer (layer dout))

(defmethod backward-layer ((layer dense-layer) dout)
  (let* ((W (layer-weights layer))
         (x (layer-last-input layer))
         (z (layer-last-z layer))
         (act (layer-activation layer))

         ;; dL/dz = dL/dout * activation'(z) (element par element)
         (dz (cond
               ;; cas ReLU
               ((eq act #'relu-vec)
                (vec-hadamard dout (relu-deriv-vec z)))
               ;; cas lineaire (pas d'activation)
               (t
                dout)))
         ;; gradients de la couche
         (grad-W (outer-product dz x))
         (grad-b dz)

         ;; gradient a renvoyer a la couche precedente
         (WT (mat-transpose W))
         (din (mat-vec-mul WT dz)))
    (setf (layer-grad-weights layer) grad-W
          (layer-grad-bias layer) grad-b)
    din))

(defgeneric backward (network dloss-dy))

(defmethod backward ((network mlp) dloss-dy)
  (let ((grad dloss-dy))
    (dolist (layer (reverse (mlp-layers network)) grad)
      (setf grad (backward-layer layer grad)))))

;;----------------------------------
;; Loss
;;----------------------------------

(defun mse-loss-and-grad (y-hat y)
  "Retourne 2 valeurs: (loss dL/dy-hat)"
  (let* ((diff (vec-sub y-hat y))  ; diff = y-hat - y
         (squared (vec-hadamard diff diff))
         (loss (* 0.5 (reduce #'+ squared))))
    (values loss diff)))         ; dL/dy-hat = y-hat - y

;;----------------------------------
;; Update gradient
;;----------------------------------

(defun apply-gradients! (layer learning-rate)
  (let ((W (layer-weights layer))
        (b (layer-bias layer))
        (gW (layer-grad-weights layer))
        (gB (layer-grad-bias layer)))
    ;; W := W - lr * gW
    (dotimes (i (length W))
      (dotimes (j (length (aref W i)))
        (setf (aref (aref W i) j)
              (- (aref (aref W i) j)
                 (* learning-rate
                    (aref (aref gW i) j)))))
      )
    ;; b := b - lr * gB
    (dotimes (i (length b))
      (setf (aref b i)
            (- (aref b i)
               (* learning-rate (aref gB i)))))
    layer))

(defun apply-gradients-network! (network learning-rate)
  (dolist (layer (mlp-layers network))
    (apply-gradients! layer learning-rate)))

;;----------------------------------
;; exemple
;;----------------------------------

(defparameter *hidden-layer*
  (make-instance 'dense-layer
                 :weights #(#(0.5 -0.2 0.3 0.1)
                            #(-0.3 0.8 0.2 0.4))
                 :bias #(0.0 0.1)
                 :activation #'relu-vec))

(defparameter *output-layer*
  (make-instance 'dense-layer
                 :weights #(#(1.0 0.5)
                            #(-0.5 1.0))
                 :bias #(0.1 0.2)
                 :activation nil)) ; lineaire

(defparameter *net*
  (make-instance 'mlp
                 :layers (list *hidden-layer* *output-layer*)))

(defparameter *x* #(0.4 0.5 0.6 0.3))

(defparameter *y* #(0.5 0.4))

;; (forward *net* *x*)

(let* ((y-hat (forward *net* *x*))
       (loss nil)
       (dloss-dy nil))
  (multiple-value-setq (loss dloss-dy)
    (mse-loss-and-grad y-hat *y*))
  (format t "Loss = ~A~%" loss)
  ;; backward : remplit les gradients dans chaque couche
  (backward *net* dloss-dy))
