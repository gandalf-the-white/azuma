(in-package :azuma)

(defun hello ()
  (format t "Hello world!~%"))

(multiple-value-bind (sec min hour day month year)
    (decode-universal-time (get-universal-time))
  (format t "~4,'0D-~2,'0D-~2,'0DT~2,'0D:~2,'0D:~2,'0D"
          year month day hour min sec))

;;-----------------------------------
;; Outils
;;-----------------------------------

;; (defun dot-product (v1 v2)
;;   (reduce #'+ (map 'list #'* v1 v2)))

;; ;; return scalar
;; ;; Example
;; ;; v1 = #(1 2)
;; ;; v2 = #(3 4)
;; ;; 1 + 2 + 3 + 4 = 10

;; (defun mat-vec-mul (matrix vec)
;;   "Multiplie chaque ligne de matrix par vec."
;;   (map 'vector
;;        (lambda (row)
;;          (dot-product row vec))
;;        matrix))

;; ;; return vector
;; ;; Example:
;; ;; |1 2| * |1| = |1*2 + 2*2| = | 5|
;; ;; |3 4|   |2|   |3*1 + 4*2|   |11|

;; (defun vec-add (v1 v2)
;;   (map 'vector #'+ v1 v2))

;; ;; return vector
;; ;; Example:
;; ;; v1 = '(1 2)
;; ;; v2 = '(3 4)
;; ;; |1 + 3| = | 4|
;; ;; |2 + 4|   | 6|

;; (defun relu (x)
;;   (max 0 x))

;; ;; return max of numbers

;; (defun relu-vec (v)
;;   (map 'vector #'relu v))

;; ;; return relu result as a vector
;; ;; (relu-vec '(4 7 2))
;; ;; #(4 7 2)

;; (defun relu-deriv-vec (z)
;;   "Renvoie un vecteur de 0/1 selon z>0"
;;   (map 'vector (lambda (x) (if (> x 0) 1 0)) z))

;; (defun vec-sub (v1 v2)
;;   (map 'vector #'- v1 v2))

;; (defun vec-hadamard (v1 v2)
;;   "Produit element par element"
;;   (map 'vector #'* v1 v2))

;; ;; exemple
;; ;; (vec-hadamard #(1 3) #(2 6))
;; ;; #(2 18)

;; (defun outer-product (v1 v2)
;;   "Retourne une matrice = v1 (colonne) * v2 (ligne)"
;;   (let ((n1 (length v1))
;;         (n2 (length v2)))
;;     (let ((m (make-array (list n1 n2))))
;;       (dotimes (i n1)
;;         (dotimes (j n2)
;;           (setf (aref m i j)
;;                 (* (aref v1 i)(aref v2 j)))))
;;       ;; On renvoie un vecteur de vecteurs pour rester coherent
;;       (map 'vector
;;            (lambda (i)
;;              (let ((row (make-array n2)))
;;                (dotimes (j n2)
;;                  (setf (aref row j) (aref m i j)))
;;                row))
;;            (loop for i below n1 collect i)))))

;; (defun mat-transpose (matrix)
;;   "Retourne la transposee d'une matrice"
;;   (let* ((rows (length matrix))
;;          (cols (length (aref matrix 0)))
;;          (result (make-array (list cols))))
;;     (dotimes (j cols)
;;       (let ((row (make-array rows)))
;;         (dotimes (i rows)
;;           (setf (aref row i) (aref (aref matrix i) j)))
;;         (setf (aref result j) row)))
;;     result))

;; (defun forward-mlp (x)
;;   ;; Couche cachee
;;   (let* ((z1 (vec-add (mat-vec-mul *w1* x) *b1*))
;;          (h (relu-vec z1))
;;          (z2 (vec-add (mat-vec-mul *w2* h) *b2*)))
;;     z2))

;; (defun make-like-matrix (matrix &key (initial-element 0.0))
;;   (let* ((rows (length matrix))
;;          (cols (length (aref matrix 0)))
;;          (m (make-array rows)))
;;     (dotimes (i rows)
;;       (let ((row (make-array cols)))
;;         (dotimes (j cols)
;;           (setf (aref row j) initial-element))
;;         (setf (aref m i) row)))
;;     m))

;; (defun make-like-vector (vec &key (initial-element 0.0))
;;   (let* ((n (length vec))
;;          (v (make-array n)))
;;     (dotimes (i n)
;;       (setf (aref v i) initial-element))
;;     v))

;; (defun mat-add! (a b)
;;   "A := A + B (in-place)."
;;   (dotimes (i (length a))
;;     (dotimes (j (length (aref a i)))
;;       (incf (aref (aref a i) j)
;;             (aref (aref b i) j))))
;;   a)

;; (defun vec-add! (a b)
;;   "A := A + B (in-place)."
;;   (dotimes (i (length a))
;;     (incf (aref a i) (aref b i)))
;;   a)

;; (defun mat-scale! (a factor)
;;   "A := factor * A (in-place)."
;;   (dotimes (i (length a))
;;     (dotimes (j (length (aref a i)))
;;       (setf (aref (aref a i) j)
;;             (* factor (aref (aref a i) j)))))
;;   a)

;; (defun vec-scale! (a factor)
;;   "A := factor * A (in-place)."
;;   (dotimes (i (length a))
;;     (setf (aref a i) (* factor (aref a i))))
;;   a)

;; (defun make-index-vector (n)
;;   (let ((v (make-array n)))
;;     (dotimes (i n)
;;       (setf (aref v i) i))
;;     v))

;; (defun shuffle-vector! (v)
;;   "Shuffle in-place (Fisher–Yates)."
;;   (let ((n (length v)))
;;     (dotimes (i n)
;;       (let* ((j (+ i (random (- n i))))
;;              (tmp (aref v i)))
;;         (setf (aref v i) (aref v j)
;;               (aref v j) tmp))))
;;   v)

;;-------------------------------------
;; Classe
;;-------------------------------------

;; (defclass sample ()
;;   ((timestamp
;;     :initarg :timestamp
;;     :accessor sample-timestamp
;;     :documentation "Timestamp (string ou nombre, p.ex. epoch ou ISO8601).")
;;    (cpu
;;     :initarg :cpu
;;     :accessor sample-cpu
;;     :documentation "Utilisation CPU normalisée (0.0 - 1.0).")
;;    (mem
;;     :initarg :mem
;;     :accessor sample-mem
;;     :documentation "Utilisation MEM normalisée (0.0 - 1.0).")))

;; (defclass dense-layer ()
;;   ((weights
;;     :initarg :weights
;;     :accessor layer-weights
;;     :documentation "Matrice des poids (vecteur de vecteurs)")
;;    (bias
;;     :initarg :bias
;;     :accessor layer-bias
;;     :documentation "Vecteur de biais")
;;    (activation
;;     :initarg :activation
;;     :accessor layer-activation
;;     :documentation "Fonction d'activation ou NIL pour lineaire")
;;    (last-input
;;     :initarg :last-input
;;     :accessor layer-last-input
;;     :initform nil
;;     :documentation "Entree de la couche")
;;    (last-z
;;     :accessor layer-last-z
;;     :initform nil
;;     :documentation "Sortie lineaire avant activation")
;;    (last-output
;;     :accessor layer-last-output
;;     :initform nil
;;     :documentation "Sortie apres activation")
;;    (grad-weights
;;     :accessor layer-grad-weights
;;     :initform nil
;;     :documentation "Gradients accumulees pour cette passe")
;;    (grad-bias
;;     :accessor layer-grad-bias
;;     :initform nil
;;     :documentation "Biais accumules pour cette passe")
;;    (accum-grad-weights
;;     :accessor layer-accum-grad-weights
;;     :initform nil
;;     :documentation "Accumulation des poids")
;;    (accum-grad-bias
;;     :accessor layer-accum-grad-bias
;;     :initform nil
;;     :documentation "Accumulation des biais")))

;; (defgeneric forward-layer (layer input))

;; (defmethod forward-layer ((layer dense-layer) input)
;;   (let* ((z (vec-add (mat-vec-mul (layer-weights layer) input)
;;                      (layer-bias layer)))
;;          (act (layer-activation layer))
;;          (out (if act
;;                   (funcall act z)
;;                   z)))
;;     (setf (layer-last-input layer) input
;;           (layer-last-z layer) z
;;           (layer-last-output layer) out)
;;     out))

;; (defclass mlp ()
;;   ((layers
;;     :initarg :layers
;;     :accessor mlp-layers
;;     :documentation "Liste de couches (instance de dense-layer)")))

;;----------------------------------
;; Forward 
;;----------------------------------

;; (defgeneric forward (network input))

;; (defmethod forward ((network mlp) input)
;;   (reduce (lambda (x layer)
;;             (forward-layer layer x))
;;           (mlp-layers network)
;;           :initial-value input))

;;----------------------------------
;; Backward 
;;----------------------------------

;; (defgeneric backward-layer (layer dout))

;; (defmethod backward-layer ((layer dense-layer) dout)
;;   (let* ((W (layer-weights layer))
;;          (x (layer-last-input layer))
;;          (z (layer-last-z layer))
;;          (act (layer-activation layer))

;;          ;; dL/dz = dL/dout * activation'(z) (element par element)
;;          (dz (cond
;;                ;; cas ReLU
;;                ((eq act #'relu-vec)
;;                 (vec-hadamard dout (relu-deriv-vec z)))
;;                ;; cas lineaire (pas d'activation)
;;                (t
;;                 dout)))
;;          ;; gradients de la couche
;;          (grad-W (outer-product dz x))
;;          (grad-b dz)

;;          ;; gradient a renvoyer a la couche precedente
;;          (WT (mat-transpose W))
;;          (din (mat-vec-mul WT dz)))
;;     (setf (layer-grad-weights layer) grad-W
;;           (layer-grad-bias layer) grad-b)
;;     din))

;; (defgeneric backward (network dloss-dy))

;; (defmethod backward ((network mlp) dloss-dy)
;;   (let ((grad dloss-dy))
;;     (dolist (layer (reverse (mlp-layers network)) grad)
;;       (setf grad (backward-layer layer grad)))))

;;----------------------------------
;; Loss
;;----------------------------------

;; (defun mse-loss-and-grad (y-hat y)
;;   "Retourne 2 valeurs: (loss dL/dy-hat)"
;;   (let* ((diff (vec-sub y-hat y))  ; diff = y-hat - y
;;          (squared (vec-hadamard diff diff))
;;          (loss (* 0.5 (reduce #'+ squared))))
;;     (values loss diff)))         ; dL/dy-hat = y-hat - y

;;----------------------------------
;; Update gradient
;;----------------------------------

;; (defun apply-gradients! (layer learning-rate)
;;   (let ((W (layer-weights layer))
;;         (b (layer-bias layer))
;;         (gW (layer-grad-weights layer))
;;         (gB (layer-grad-bias layer)))
;;     ;; W := W - lr * gW
;;     (dotimes (i (length W))
;;       (dotimes (j (length (aref W i)))
;;         (setf (aref (aref W i) j)
;;               (- (aref (aref W i) j)
;;                  (* learning-rate
;;                     (aref (aref gW i) j)))))
;;       )
;;     ;; b := b - lr * gB
;;     (dotimes (i (length b))
;;       (setf (aref b i)
;;             (- (aref b i)
;;                (* learning-rate (aref gB i)))))
;;     layer))

;; (defun apply-gradients-network! (network learning-rate)
;;   (dolist (layer (mlp-layers network))
;;     (apply-gradients! layer learning-rate)))

;;----------------------------------
;; accu gradients
;;----------------------------------

;; (defun zero-accum-gradients-network! (network)
;;   (dolist (layer (mlp-layers network))
;;     (let ((W (layer-weights layer))
;;           (b (layer-bias layer)))
;;       (setf (layer-accum-grad-weights layer)
;;             (make-like-matrix W :initial-element 0.0)
;;             (layer-accum-grad-bias layer)
;;             (make-like-vector b :initial-element 0.0)))))

;; (defun accumulate-current-gradients! (network)
;;   (dolist (layer (mlp-layers network))
;;     (let ((gW (layer-grad-weights layer))
;;           (gB (layer-grad-bias layer))
;;           (aW (layer-accum-grad-weights layer))
;;           (aB (layer-accum-grad-bias layer)))
;;       ;; aW += gW ; aB += gB
;;       (mat-add! aW gW)
;;       (vec-add! aB gB))))

;; (defun average-accum-gradients-network! (network batch-size)
;;   (let ((factor (/ 1.0 batch-size)))
;;     (dolist (layer (mlp-layers network))
;;       (let ((aW (layer-accum-grad-weights layer))
;;             (aB (layer-accum-grad-bias layer)))
;;         (mat-scale! aW factor)
;;         (vec-scale! aB factor)
;;         ;; on met ces moyennes dans grad-weights / grad-bias
;;         (setf (layer-grad-weights layer) aW
;;               (layer-grad-bias layer)    aB)))))

;;----------------------------------
;; training
;;----------------------------------

(defun train-one-step (net x y learning-rate)
  (let* ((y-hat (forward net x))
         (loss  nil)
         (dloss-dy nil))
    (multiple-value-setq (loss dloss-dy)
      (mse-loss-and-grad y-hat y))
    (format t "Loss avant update: ~A~%" loss)
    (backward net dloss-dy)
    (apply-gradients-network! net learning-rate)
    (let* ((y-hat-new (forward net x))
           (loss-new  (multiple-value-bind (l _)
                          (mse-loss-and-grad y-hat-new y)
                        (declare (ignore _))
                        l)))
      (format t "Loss après update: ~A~%" loss-new))))

(defun train-one-batch (net xs ys learning-rate)
  (let* ((batch-size (length xs))
         (total-loss 0.0))
    ;; 1) zéro des accumulateurs
    (zero-accum-gradients-network! net)

    ;; 2) boucle sur les exemples
    (dotimes (n batch-size)
      (let* ((x (elt xs n))
             (y (elt ys n))
             (y-hat (forward net x))
             (loss nil)
             (dloss-dy nil))
        (multiple-value-setq (loss dloss-dy)
          (mse-loss-and-grad y-hat y))
        (incf total-loss loss)
        ;; backward pour cet exemple
        (backward net dloss-dy)
        ;; accumuler les gradients de cet exemple
        (accumulate-current-gradients! net)))

    ;; 3) moyenne des gradients sur le batch
    (average-accum-gradients-network! net batch-size)

    ;; 4) update des poids avec les gradients moyens
    (apply-gradients-network! net learning-rate)

    ;; 5) renvoyer la loss moyenne du batch
    (/ total-loss batch-size)))

;; (defun train-mini-batch (net xs ys index-vector start batch-size learning-rate)
;;   "Entraîne sur un mini-batch défini par INDEX-VECTOR[start .. start+batch-size[."
;;   (let ((total-loss 0.0))
;;     ;; 1) zéro des accumulateurs
;;     (zero-accum-gradients-network! net)

;;     ;; 2) boucle sur les exemples de ce batch
;;     (dotimes (k batch-size)
;;       (let* ((idx (aref index-vector (+ start k)))
;;              (x   (aref xs idx))
;;              (y   (aref ys idx))
;;              (y-hat (forward net x))
;;              (loss nil)
;;              (dloss-dy nil))
;;         (multiple-value-setq (loss dloss-dy)
;;           (mse-loss-and-grad y-hat y))
;;         (incf total-loss loss)
;;         ;; backward + accumulation
;;         (backward net dloss-dy)
;;         (accumulate-current-gradients! net)))

;;     ;; 3) moyenne des gradients sur ce batch
;;     (average-accum-gradients-network! net batch-size)

;;     ;; 4) update des poids
;;     (apply-gradients-network! net learning-rate)

;;     ;; 5) renvoie la loss moyenne du batch
;;     (/ total-loss batch-size)))

;;----------------------------------
;; epochs
;;----------------------------------

;; (defun train-epochs (net xs ys
;;                      &key
;;                        (epochs 10)
;;                        (batch-size 2)
;;                        (learning-rate 0.01)
;;                        (verbose t))
;;   "Retourne une liste de (epoch avg-loss) pour tracer ensuite."
;;   (let* ((n-samples (length xs)))
;;     (assert (= n-samples (length ys)))
;;     (let ((indices (make-index-vector n-samples))
;;           (loss-history '()))  ;; liste de (epoch avg-loss)
;;       (dotimes (epoch epochs loss-history)
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
;;               (format t "Epoch ~D, loss moyenne = ~A~%" epoch avg-loss))
;;             ;; enregistrer (epoch avg-loss)
;;             (push (list epoch avg-loss) loss-history))))
;;       ;; on a push dans l'ordre inverse, on remet dans le bon sens
;;       (nreverse loss-history))))

;;----------------------------------
;; gnuplot
;;----------------------------------

;; (defun export-time-series-to-dat (sample filepath)
;;   "Ecrire un fichier texte: index cpu mem"
;;   (with-open-file (out filepath
;;                        :direction :output
;;                        :if-exists :supersede
;;                        :if-does-not-exist :create)
;;     (dotimes (i (length sample))
;;       (let* ((s (aref sample i))
;;              (cpu (sample-cpu s))
;;              (mem (sample-mem s)))
;;         (format out "~D ~F ~F~%" i cpu mem)))))

;; (defun export-time-series-to-time-dat (samples filepath)
;;   "Écrit un fichier texte: timestamp cpu mem (timestamp en string)."
;;   (with-open-file (out filepath
;;                        :direction :output
;;                        :if-exists :supersede
;;                        :if-does-not-exist :create)
;;     (dotimes (i (length samples))
;;       (let* ((s   (aref samples i))
;;              (ts  (sample-timestamp s))
;;              (cpu (sample-cpu s))
;;              (mem (sample-mem s)))
;;         (format out "~A ~F ~F~%" ts cpu mem)))))

;; (defun export-loss-history-to-dat (loss-history filepath)
;;   "LOSS-HISTORY est une liste de (epoch avg-loss)."
;;   (with-open-file (out filepath
;;                        :direction :output
;;                        :if-exists :supersede
;;                        :if-does-not-exist :create)
;;     (dolist (entry loss-history)
;;       (destructuring-bind (epoch loss) entry
;;         (format out "~D ~F~%" epoch loss)))))

;;----------------------------------
;; exemple
;;----------------------------------

(defparameter *x1* #(0.4 0.5 0.6 0.3))
(defparameter *y1* #(0.5 0.4))

(defparameter *x2* #(0.2 0.1 0.7 0.9))
(defparameter *y2* #(0.8 0.2))

(defparameter *xs* (vector *x1* *x2*))
(defparameter *ys* (vector *y1* *y2*))

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

(defparameter *W1* #(#(0.5 -0.2 0.3 0.1) #(-0.3 0.8 0.2 0.4)))
(defparameter *b1* #(0.0 0.1))
(defparameter *W2* #(#(1.0 0.5) #(-0.5 1.0)))
(defparameter *b2* #(0.1 0.2))

(defparameter *time-series*
  (vector
   (make-instance 'sample :timestamp "2025-11-26T10:00:00" :cpu 0.62 :mem 0.47)
   (make-instance 'sample :timestamp "2025-11-26T11:00:00" :cpu 0.70 :mem 0.50)
   (make-instance 'sample :timestamp "2025-11-26T12:00:00" :cpu 0.55 :mem 0.40)))

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

;; (forward *net* *x*)

;; (let* ((y-hat (forward *net* *x*))
;;        (loss nil)
;;        (dloss-dy nil))
;;   (multiple-value-setq (loss dloss-dy)
;;     (mse-loss-and-grad y-hat *y*))
;;   (format t "Loss = ~A~%" loss)
;;   ;; backward : remplit les gradients dans chaque couche
;;   (backward *net* dloss-dy))

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

;; (train-epochs *net* *xs* *ys*
;;               :epochs 100
;;               :batch-size 2
;;               :learning-rate 0.05
;;               :verbose t)

;; (defparameter *loss-history*
;;   (train-epochs *net* *xs* *ys*
;;                 :epochs 100
;;                 :batch-size 2
;;                 :learning-rate 0.05
;;                 :verbose t))
