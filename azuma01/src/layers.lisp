(in-package :azuma)

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
    :documentation "Biais accumules pour cette passe")
   (accum-grad-weights
    :accessor layer-accum-grad-weights
    :initform nil
    :documentation "Accumulation des poids")
   (accum-grad-bias
    :accessor layer-accum-grad-bias
    :initform nil
    :documentation "Accumulation des biais")))

(defgeneric forward-layer (layer input))

(defmethod forward-layer ((layer dense-layer) input)
  (let* ((z (vec-add (mat-vec-mul (layer-weights layer) input)
                     (layer-bias layer)))
         (act (layer-activation layer))
         (out (if act
                  (funcall act z)
                  z)))
    (setf (layer-last-input layer) input
          (layer-last-z layer) z
          (layer-last-output layer) out)
    out))

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
