(in-package :asdf-user)

(defsystem "azuma"
  :version "0.0.1"
  :author "spike spiegel"
  :license "MIT"
  :depends-on ()
  :components ((:module "src"
                :components
                ((:file "package")
                 (:file "utils")
                 (:file "layers")
                 (:file "network")
                 (:file "loss")
                 (:file "training")
                 (:file "data-export")
                 (:file "rnn")
                 (:file "rnn-training"))))
  :description ""
  :in-order-to ((test-op (test-op "azuma/tests"))))

(defsystem "azuma/tests"
  :author ""
  :license ""
  :depends-on ("azuma"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for azuma"
  :perform (test-op (op c) (symbol-call :rove :run c)))
