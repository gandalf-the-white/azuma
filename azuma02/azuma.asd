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
                 (:file "training"))))
  :description "")
