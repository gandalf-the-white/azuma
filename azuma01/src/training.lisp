(in-package :azuma)

(defun train-mini-batch (net xs ys index-vector start batch-size learning-rate)
  "Entraîne sur un mini-batch défini par INDEX-VECTOR[start .. start+batch-size[."
  (let ((total-loss 0.0))
    ;; 1) zéro des accumulateurs
    (zero-accum-gradients-network! net)

    ;; 2) boucle sur les exemples de ce batch
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

    ;; 3) moyenne des gradients sur ce batch
    (average-accum-gradients-network! net batch-size)

    ;; 4) update des poids
    (apply-gradients-network! net learning-rate)

    ;; 5) renvoie la loss moyenne du batch
    (/ total-loss batch-size)))

(defun train-epochs (net xs ys
                     &key
                       (epochs 10)
                       (batch-size 2)
                       (learning-rate 0.01)
                       (verbose t))
  "Retourne une liste de (epoch avg-loss) pour tracer ensuite."
  (let* ((n-samples (length xs)))
    (assert (= n-samples (length ys)))
    (let ((indices (make-index-vector n-samples))
          (loss-history '()))  ;; liste de (epoch avg-loss)
      (dotimes (epoch epochs loss-history)
        ;; shuffle des indices à chaque epoch
        (shuffle-vector! indices)
        (let ((epoch-loss 0.0)
              (n-batches 0))
          ;; boucle sur les mini-batches
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
              (format t "Epoch ~D, loss moyenne = ~A~%" epoch avg-loss))
            ;; enregistrer (epoch avg-loss)
            (push (list epoch avg-loss) loss-history))))
      ;; on a push dans l'ordre inverse, on remet dans le bon sens
      (nreverse loss-history))))

(defun export-loss-history-to-dat (loss-history filepath)
  "LOSS-HISTORY est une liste de (epoch avg-loss)."
  (with-open-file (out filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (dolist (entry loss-history)
      (destructuring-bind (epoch loss) entry
        (format out "~D ~F~%" epoch loss)))))
