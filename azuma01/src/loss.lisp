(in-package :azuma)

(defun mse-loss-and-grad (y-hat y)
  "Retourne 2 valeurs: (loss dL/dy-hat)"
  (let* ((diff (vec-sub y-hat y))  ; diff = y-hat - y
         (squared (vec-hadamard diff diff))
         (loss (* 0.5 (reduce #'+ squared))))
    (values loss diff)))         ; dL/dy-hat = y-hat - y
