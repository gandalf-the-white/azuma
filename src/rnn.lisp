(in-package :azuma)

;;;;========================================================
;;;; RNN cell + réseau RNN many-to-one
;;;;========================================================

(defclass rnn-cell ()
  ((Wx  :initarg :Wx  :accessor rnn-Wx)   ; (hidden-size x input-size)
   (Wh  :initarg :Wh  :accessor rnn-Wh)   ; (hidden-size x hidden-size)
   (bh  :initarg :bh  :accessor rnn-bh)   ; (hidden-size)
   (activation
    :initarg :activation
    :accessor rnn-activation
    :initform #'tanh)

   ;; caches pour BPTT sur la dernière séquence
   (last-inputs   :accessor rnn-last-inputs   :initform nil) ; vector de x_t
   (last-hiddens  :accessor rnn-last-hiddens  :initform nil) ; vector de h_t (0..T)
   (last-T        :accessor rnn-last-T        :initform 0)

   ;; gradients pour 1 séquence
   (grad-Wx  :accessor rnn-grad-Wx  :initform nil)
   (grad-Wh  :accessor rnn-grad-Wh  :initform nil)
   (grad-bh  :accessor rnn-grad-bh  :initform nil)))

(defun make-zero-vector (n)
  (let ((v (make-array n)))
    (dotimes (i n) (setf (aref v i) 0.0))
    v))

(defun tanh-vec (v)
  (map 'vector #'tanh v))

(defun tanh-deriv-from-output (h)
  "Pour tanh, h = tanh(z) donc d/dz = 1 - h^2."
  (map 'vector (lambda (x) (- 1.0 (* x x))) h))

(defun rnn-cell-forward-seq (cell inputs)
  "inputs = vector de x_t (vecteurs). Retourne h_T."
  (let* ((Wx (rnn-Wx cell))
         (Wh (rnn-Wh cell))
         (bh (rnn-bh cell))
         (TT  (length inputs))
         (hidden-size (length bh))
         ;; h_0 = 0
         (h0 (make-zero-vector hidden-size))
         (hiddens (make-array (1+ TT))) ; h_0..h_T
         )
    (setf (aref hiddens 0) h0)
    ;; forward sur t = 1..TT
    (dotimes (tt TT)
      (let* ((x_t (aref inputs tt))
             (h-prev (aref hiddens tt))
             (Wx_x  (mat-vec-mul Wx x_t))   ; hidden-size
             (Wh_h  (mat-vec-mul Wh h-prev)) ; hidden-size
             (z     (vec-add (vec-add Wx_x Wh_h) bh))
             (h     (tanh-vec z)))
        (setf (aref hiddens (1+ tt)) h)))
    ;; stocker les caches
    (setf (rnn-last-inputs cell)  inputs
          (rnn-last-hiddens cell) hiddens
          (rnn-last-T cell)       TT)
    ;; retourner le dernier état caché
    (aref hiddens TT)))

(defun rnn-cell-backward-seq (cell dL-dh-T)
  "BPTT pour 1 séquence.
Retourne NIL, mais remplit rnn-grad-Wx, rnn-grad-Wh, rnn-grad-bh
et renvoie aussi le gradient par rapport aux entrées (facultatif)."
  (let* ((Wx (rnn-Wx cell))
         (Wh (rnn-Wh cell))
         (bh (rnn-bh cell))
         (inputs  (rnn-last-inputs cell))     ; x_t
         (hiddens (rnn-last-hiddens cell))    ; h_0..h_T
         (TT       (rnn-last-T cell))
         (hidden-size (length bh))
         (input-size (length (aref inputs 0)))
         ;; init gradients à 0
         (gWx (make-like-matrix Wx :initial-element 0.0))
         (gWh (make-like-matrix Wh :initial-element 0.0))
         (gbh (make-like-vector  bh :initial-element 0.0))
         ;; gradient w.r.t h_t courant (commence à TT)
         (dL-dh-next dL-dh-T)
         ;; éventuellement gradient w.r.t inputs
         (dL-dxs (make-array TT)))
    ;; boucle backward t = T..1
    (loop for tt downfrom TT above 0 do
      (let* ((x_t   (aref inputs (1- tt)))  ; attention indices
             (h_t   (aref hiddens tt))
             (h_tm1 (aref hiddens (1- tt))) ; h_{tt-1}
             ;; dL/dh_t = dL_dh_next (accumulé)
             (dL-dh dL-dh-next)
             ;; dL/dz_t = dL/dh_t * (1 - h_t^2)
             (dh-dz (tanh-deriv-from-output h_t))
             (dL-dz (vec-hadamard dL-dh dh-dz)))
        ;; grad Wx += dL/dz_t * x_t^TT
        (mat-add! gWx (outer-product dL-dz x_t))
        ;; grad Wh += dL/dz_t * h_{t-1}^TT
        (mat-add! gWh (outer-product dL-dz h_tm1))
        ;; grad bh += dL/dz_t
        (vec-add! gbh dL-dz)
        ;; gradients pour étape précédente
        (let* ((WhT (mat-transpose Wh))
               (dL-dh-prev (mat-vec-mul WhT dL-dz))
               ;; dL/dx_t (si tu veux le garder)
               (WxT (mat-transpose Wx))
               (dL-dx (mat-vec-mul WxT dL-dz)))
          (setf (aref dL-dxs (1- tt)) dL-dx
                dL-dh-next dL-dh-prev))))
    ;; stocker les gradients
    (setf (rnn-grad-Wx cell) gWx
          (rnn-grad-Wh cell) gWh
          (rnn-grad-bh cell) gbh)
    dL-dxs))

(defun rnn-apply-gradients! (cell learning-rate)
  (let ((Wx (rnn-Wx cell))
        (Wh (rnn-Wh cell))
        (bh (rnn-bh cell))
        (gWx (rnn-grad-Wx cell))
        (gWh (rnn-grad-Wh cell))
        (gbh (rnn-grad-bh cell)))
    ;; Wx := Wx - lr * gWx
    (dotimes (i (length Wx))
      (dotimes (j (length (aref Wx i)))
        (setf (aref (aref Wx i) j)
              (- (aref (aref Wx i) j)
                 (* learning-rate (aref (aref gWx i) j))))))
    ;; Wh
    (dotimes (i (length Wh))
      (dotimes (j (length (aref Wh i)))
        (setf (aref (aref Wh i) j)
              (- (aref (aref Wh i) j)
                 (* learning-rate (aref (aref gWh i) j))))))
    ;; bh
    (dotimes (i (length bh))
      (setf (aref bh i)
            (- (aref bh i)
               (* learning-rate (aref gbh i)))))
    cell))

;;;;========================================================
;;;; Réseau RNN many-to-one : cell + couche de sortie dense
;;;;========================================================

(defclass rnn-network ()
  ((cell
    :initarg :cell
    :accessor rnn-net-cell)
   (output-layer
    :initarg :output-layer
    :accessor rnn-net-output-layer)))

(defun rnn-forward-seq (net seq-x)
  "Forward sur une séquence (vector de x_t).
Retourne y-hat."
  (let* ((cell (rnn-net-cell net))
         (out-layer (rnn-net-output-layer net))
         ;; h_T
         (h-T (rnn-cell-forward-seq cell seq-x))
         ;; couche de sortie linéaire
         (y-hat (forward-layer out-layer h-T)))
    y-hat))

(defun rnn-backward-seq (net dL-dy)
  "Backward pour une séquence.
1) backward sur couche de sortie -> dL/dh_T
2) BPTT sur la cell RNN."
  (let* ((cell (rnn-net-cell net))
         (out-layer (rnn-net-output-layer net)))
    ;; backward couche de sortie (dense-layer)
    (let ((dL-dh-T (backward-layer out-layer dL-dy)))
      ;; backward à travers le temps
      (rnn-cell-backward-seq cell dL-dh-T))))

(defun rnn-apply-gradients-network! (net learning-rate)
  "Update RNN cell + couche de sortie."
  (rnn-apply-gradients! (rnn-net-cell net) learning-rate)
  (apply-gradients! (rnn-net-output-layer net) learning-rate))

;;; petit constructeur de test : input-size 2 (CPU/MEM), hidden 4, output 2
(defun make-rnn-network-2-4-2 ()
  (let* ((input-size 2)
         (hidden-size 4)
         ;; init Wx, Wh, bh avec de petits nombres
         (Wx (vector
              #(0.1  0.2)
              #(0.0 -0.1)
              #(0.05 0.03)
              #(-0.02 0.04)))
         (Wh (vector
              #(0.1  0.0  0.0  0.0)
              #(0.0  0.1  0.0  0.0)
              #(0.0  0.0  0.1  0.0)
              #(0.0  0.0  0.0  0.1)))
         (bh (make-zero-vector hidden-size))
         (cell (make-instance 'rnn-cell
                              :Wx Wx
                              :Wh Wh
                              :bh bh
                              :activation #'tanh-vec))
         ;; couche de sortie (hidden -> 2)
         (out-layer (make-instance 'dense-layer
                                   :weights #(#(0.1 0.1 0.1 0.1)
                                              #(0.05 -0.05 0.02 -0.02))
                                   :bias    #(0.0 0.0)
                                   :activation nil)))
    (make-instance 'rnn-network
                   :cell cell
                   :output-layer out-layer)))
