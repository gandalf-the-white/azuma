(in-package :azuma)

(defun rnn-train-one-seq (net seq-x y learning-rate)
  "Entraîne le réseau RNN sur UNE séquence (SGD).
Retourne la loss."
  (let* ((y-hat (rnn-forward-seq net seq-x))
         (loss nil)
         (dL-dy nil))
    (multiple-value-setq (loss dL-dy)
      (mse-loss-and-grad y-hat y))
    ;; backward
    (rnn-backward-seq net dL-dy)
    ;; update
    (rnn-apply-gradients-network! net learning-rate)
    loss))

(defun rnn-train-epochs (net seqs-xs ys
                         &key
                           (epochs 10)
                           (learning-rate 0.01)
                           (verbose t))
  "seqs-xs : vector de séquences, chaque séquence est (vector de x_t).
ys : vector de cibles (un y par séquence).
Retourne une liste de (epoch avg-loss)."
  (let* ((n-samples (length seqs-xs)))
    (assert (= n-samples (length ys)))
    (let ((indices (make-index-vector n-samples))
          (loss-history '()))
      (dotimes (epoch epochs (nreverse loss-history))
        (shuffle-vector! indices)
        (let ((epoch-loss 0.0))
          (dotimes (i n-samples)
            (let* ((idx   (aref indices i))
                   (seq-x (aref seqs-xs idx))
                   (y     (aref ys idx))
                   (loss  (rnn-train-one-seq net seq-x y learning-rate)))
              (incf epoch-loss loss)))
          (let ((avg-loss (/ epoch-loss n-samples)))
            (when verbose
              (format t "RNN Epoch ~D, loss moyenne = ~A~%" epoch avg-loss))
            (push (list epoch avg-loss) loss-history)))))))
