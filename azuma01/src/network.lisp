(in-package :azuma)

(defclass mlp ()
  ((layers
    :initarg :layers
    :accessor mlp-layers
    :documentation "Liste de couches (instance de dense-layer)")))

(defgeneric forward (network input))

(defmethod forward ((network mlp) input)
  (reduce (lambda (x layer)
            (forward-layer layer x))
          (mlp-layers network)
          :initial-value input))

(defgeneric backward (network dloss-dy))

(defmethod backward ((network mlp) dloss-dy)
  (let ((grad dloss-dy))
    (dolist (layer (reverse (mlp-layers network)) grad)
      (setf grad (backward-layer layer grad)))))

(defun zero-accum-gradients-network! (network)
  (dolist (layer (mlp-layers network))
    (let ((W (layer-weights layer))
          (b (layer-bias layer)))
      (setf (layer-accum-grad-weights layer)
            (make-like-matrix W :initial-element 0.0)
            (layer-accum-grad-bias layer)
            (make-like-vector b :initial-element 0.0)))))

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

(defun apply-gradients-network! (network learning-rate)
  (dolist (layer (mlp-layers network))
    (apply-gradients! layer learning-rate)))

;;; Petit constructeur concret 4 -> 2 -> 2 (CPU/MEM avec k=2)

(defun make-mlp-2-4-2 ()
  (let* ((hidden (make-instance 'dense-layer
                                :weights #(#( 0.5  -0.2   0.3   0.1)
                                           #(-0.3   0.8   0.2   0.4))
                                :bias    #(0.0 0.1)
                                :activation #'relu-vec))
         (output (make-instance 'dense-layer
                                :weights #(#( 1.0   0.5)
                                           #(-0.5   1.0))
                                :bias    #(0.1 0.2)
                                :activation nil)))
    (make-instance 'mlp :layers (list hidden output))))
