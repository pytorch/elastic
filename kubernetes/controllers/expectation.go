/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package controllers

import (
	"fmt"
	v1 "github.com/kubeflow/common/pkg/apis/common/v1"
	"github.com/kubeflow/common/pkg/controller.v1/common"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func (r *ElasticJobReconciler) satisfiedExpections(job *v1alpha1.ElasticJob) bool {
	satisfied := false
	key, err := common.KeyFunc(job)
	if err != nil {
		runtime.HandleError(fmt.Errorf("couldn't get key for job object %v: %v", job, err))
		return false
	}
	for rtype := range job.Spec.ReplicaSpecs {
		// Check the expectations of the pods.
		expectationPodsKey := common.GenExpectationPodsKey(key, string(rtype))
		satisfied = satisfied || r.jobController.Expectations.SatisfiedExpectations(expectationPodsKey)
		// Check the expectations of the services.
		expectationServicesKey := common.GenExpectationServicesKey(key, string(rtype))
		satisfied = satisfied || r.jobController.Expectations.SatisfiedExpectations(expectationServicesKey)
	}

	return satisfied
}

// onDependentCreateFunc modify expectations when dependent (pod/service) creation observed.
func onDependentCreateFunc(r reconcile.Reconciler) func(event.CreateEvent) bool {
	return func(e event.CreateEvent) bool {
		ejr, ok := r.(*ElasticJobReconciler)
		if !ok {
			return true
		}

		// Reconcile any ElasticJob create Event
		if _, ok := e.Object.(*v1alpha1.ElasticJob); ok {
			return true
		}

		// Predicates are provided to filter events before they are given to the EventHandler.
		// Events will be passed to the EventHandler iff all provided Predicates evaluate to true.
		// In this case, it won't filter out pods/services whose owner are not TorchElasticJob,
		// we need to check group label to filter them out.
		value := e.Meta.GetLabels()[v1.GroupNameLabel]
		if value == "" || value != ejr.GetAPIGroupVersion().Group {
			return false
		}

		ejr.Log.Info(fmt.Sprintf("Update on create function: %s create object %s", ejr.ControllerName(), e.Meta.GetName()))
		rtype := e.Meta.GetLabels()[v1.ReplicaTypeLabel]
		if len(rtype) == 0 {
			return false
		}

		if controllerRef := metav1.GetControllerOf(e.Meta); controllerRef != nil {
			var expectKey string
			if _, ok := e.Object.(*corev1.Pod); ok {
				expectKey = common.GenExpectationPodsKey(e.Meta.GetNamespace()+"/"+controllerRef.Name, rtype)
			}

			if _, ok := e.Object.(*corev1.Service); ok {
				expectKey = common.GenExpectationServicesKey(e.Meta.GetNamespace()+"/"+controllerRef.Name, rtype)
			}

			ejr.jobController.Expectations.CreationObserved(expectKey)
			return true
		}

		return true
	}
}

// onDependentDeleteFunc listens on elasticJob deletion and also modify expectations when dependent (pod/service) deletion observed.
func onDependentDeleteFunc(r reconcile.Reconciler) func(event.DeleteEvent) bool {
	return func(e event.DeleteEvent) bool {
		ejr, ok := r.(*ElasticJobReconciler)
		if !ok {
			return true
		}

		// Reconcile any ElasticJob delete Event
		if _, ok := e.Object.(*v1alpha1.ElasticJob); ok {
			return true
		}

		// Predicates are provided to filter events before they are given to the EventHandler.
		// Events will be passed to the EventHandler iff all provided Predicates evaluate to true.
		// In this case, it won't filter out pods/services whose owner are not ElasticJob,
		// we need to check group label to filter them out.
		value := e.Meta.GetLabels()[v1.GroupNameLabel]
		if value == "" || value != ejr.GetAPIGroupVersion().Group {
			return false
		}

		ejr.Log.Info(fmt.Sprintf("Update on delete function: %s create object %s", ejr.ControllerName(), e.Meta.GetName()))
		rtype := e.Meta.GetLabels()[v1.ReplicaTypeLabel]
		if len(rtype) == 0 {
			return false
		}

		if controllerRef := metav1.GetControllerOf(e.Meta); controllerRef != nil {
			var expectKey string
			if _, ok := e.Object.(*corev1.Pod); ok {
				expectKey = common.GenExpectationPodsKey(e.Meta.GetNamespace()+"/"+controllerRef.Name, rtype)
			}

			if _, ok := e.Object.(*corev1.Service); ok {
				expectKey = common.GenExpectationServicesKey(e.Meta.GetNamespace()+"/"+controllerRef.Name, rtype)
			}
			ejr.jobController.Expectations.DeleteExpectations(expectKey)
			return true
		}

		return true
	}
}
