(in-package :azuma)

(defclass sample ()
  ((timestamp
    :initarg :timestamp
    :accessor sample-timestamp
    :documentation "Timestamp (string ou nombre, p.ex. epoch ou ISO8601).")
   (cpu
    :initarg :cpu
    :accessor sample-cpu
    :documentation "Utilisation CPU normalisée (0.0 - 1.0).")
   (mem
    :initarg :mem
    :accessor sample-mem
    :documentation "Utilisation MEM normalisée (0.0 - 1.0).")))

(defun export-time-series-to-dat (sample filepath)
  "Ecrire un fichier texte: index cpu mem"
  (with-open-file (out filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (dotimes (i (length sample))
      (let* ((s (aref sample i))
             (cpu (sample-cpu s))
             (mem (sample-mem s)))
        (format out "~D ~F ~F~%" i cpu mem)))))

(defun export-time-series-to-time-dat (samples filepath)
  "Écrit un fichier texte: timestamp cpu mem (timestamp en string)."
  (with-open-file (out filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (dotimes (i (length samples))
      (let* ((s   (aref samples i))
             (ts  (sample-timestamp s))
             (cpu (sample-cpu s))
             (mem (sample-mem s)))
        (format out "~A ~F ~F~%" ts cpu mem)))))
