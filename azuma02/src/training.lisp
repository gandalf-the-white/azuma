(in-package :azuma)

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

(defun export-loss-history-to-dat (loss-history filepath)
  "LOSS-HISTORY list of (epoch avg-loss)."
  (with-open-file (out filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (dolist (entry loss-history)
      (destructuring-bind (epoch loss) entry
        (format out "~D ~F~%" epoch loss)))))
