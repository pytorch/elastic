/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package v1alpha1

import (
	common "github.com/kubeflow/common/pkg/apis/common/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// +kubebuilder:printcolumn:name="Min",type=integer,JSONPath=`.spec.minReplicas`
// +kubebuilder:printcolumn:name="Max",type=integer,JSONPath=`.spec.maxReplicas`
// +kubebuilder:printcolumn:name="Desired",type=integer,JSONPath=`.spec.replicaSpecs[Worker].replicas`
// +kubebuilder:printcolumn:name="rdzvEndpoint",type=string,JSONPath=`.spec.rdzvEndpoint`
// ElasticJobSpec defines the desired state of ElasticJob
type ElasticJobSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	RunPolicy common.RunPolicy `json:",inline"`

	// +kubebuilder:validation:MinItems=1
	ReplicaSpecs map[common.ReplicaType]*common.ReplicaSpec `json:"replicaSpecs"`
	RdzvEndpoint string                                     `json:"rdzvEndpoint"`

	// +kubebuilder:validation:Minimum=1
	MinReplicas *int32 `json:"minReplicas,omitempty"`
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`
}

// ElasticJobStatus defines the observed state of ElasticJob
type ElasticJobStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	common.JobStatus `json:",inline"`
}

// +kubebuilder:object:root=true

// ElasticJob is the Schema for the elasticjobs API
type ElasticJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ElasticJobSpec   `json:"spec,omitempty"`
	Status ElasticJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ElasticJobList contains a list of ElasticJob
type ElasticJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ElasticJob `json:"items"`
}

type ElasticJobReplicaType common.ReplicaType

const (
	// ElasticReplicaTypeEtcd is the type for etcd of Elastic Job.
	ElasticReplicaTypeEtcd ElasticJobReplicaType = "Etcd"

	// ElasticReplicaTypeWorker is the type for workers of Elastic Job.
	ElasticReplicaTypeWorker ElasticJobReplicaType = "Worker"
)

func init() {
	SchemeBuilder.Register(&ElasticJob{}, &ElasticJobList{})
}
