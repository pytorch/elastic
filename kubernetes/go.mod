module github.com/pytorch/elastic/kubernetes

go 1.13

require (
	github.com/go-logr/logr v0.1.0
	github.com/kubeflow/common v0.0.0-20200313171840-64f943084a05
	github.com/onsi/ginkgo v1.8.0
	github.com/onsi/gomega v1.5.0
	k8s.io/api v0.15.10
	k8s.io/apimachinery v0.15.10
	k8s.io/client-go v0.15.10
	k8s.io/kubernetes v1.15.10
	sigs.k8s.io/controller-runtime v0.3.0
)

replace (
	k8s.io/api => k8s.io/api v0.15.10
	k8s.io/apiextensions-apiserver => k8s.io/apiextensions-apiserver v0.15.10
	k8s.io/apimachinery => k8s.io/apimachinery v0.15.11-beta.0
	k8s.io/apiserver => k8s.io/apiserver v0.15.10
	k8s.io/cli-runtime => k8s.io/cli-runtime v0.15.10
	k8s.io/client-go => k8s.io/client-go v0.15.10
	k8s.io/cloud-provider => k8s.io/cloud-provider v0.15.10
	k8s.io/cluster-bootstrap => k8s.io/cluster-bootstrap v0.15.10
	k8s.io/code-generator => k8s.io/code-generator v0.15.10
	k8s.io/component-base => k8s.io/component-base v0.15.10
	k8s.io/cri-api => k8s.io/cri-api v0.15.11-beta.0
	k8s.io/csi-translation-lib => k8s.io/csi-translation-lib v0.15.10
	k8s.io/kube-aggregator => k8s.io/kube-aggregator v0.15.10
	k8s.io/kube-controller-manager => k8s.io/kube-controller-manager v0.15.10
	k8s.io/kube-proxy => k8s.io/kube-proxy v0.15.10
	k8s.io/kube-scheduler => k8s.io/kube-scheduler v0.15.10
	k8s.io/kubectl => k8s.io/kubectl v0.15.11-beta.0
	k8s.io/kubelet => k8s.io/kubelet v0.15.10
	k8s.io/legacy-cloud-providers => k8s.io/legacy-cloud-providers v0.15.10
	k8s.io/metrics => k8s.io/metrics v0.15.10
	k8s.io/node-api => k8s.io/node-api v0.15.10
	k8s.io/sample-apiserver => k8s.io/sample-apiserver v0.15.10
	k8s.io/sample-cli-plugin => k8s.io/sample-cli-plugin v0.15.10
	k8s.io/sample-controller => k8s.io/sample-controller v0.15.10
)
