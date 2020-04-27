/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package controllers

import (
	"context"
	"github.com/go-logr/logr"
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	"github.com/kubeflow/common/pkg/controller.v1/common"
	logger "github.com/kubeflow/common/pkg/util"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/workqueue"
	k8scontroller "k8s.io/kubernetes/pkg/controller"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

const (
	controllerName      = "elastic-job-controller"
	elasticJobRoleLabel = "elastic-job-role"
)

var (
	jobOwnerKey           = ".metadata.controller"
	defaultTTLseconds     = int32(100)
	defaultCleanPodPolicy = commonv1.CleanPodPolicyNone
)

// ElasticJobReconciler reconciles a ElasticJob object
type ElasticJobReconciler struct {
	jobController common.JobController
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// Reconcile reads that state of the cluster for a ElasticJob object and makes changes based on the state read
// and what is in the ElasticJob.Spec
// Automatically generate RBAC rules to allow the Controller to read and write Deployments
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=elastic.pytorch.org,resources=elasticjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=elastic.pytorch.org,resources=elasticjobs/status,verbs=get;update;patch

func (r *ElasticJobReconciler) Reconcile(req ctrl.Request) (ctrl.Result, error) {
	// Fetch the ElasticJob instance
	elasticJob := &v1alpha1.ElasticJob{}
	err := r.Get(context.TODO(), req.NamespacedName, elasticJob)
	if err != nil {
		if errors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			return reconcile.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return reconcile.Result{}, err
	}

	log := logger.LoggerForJob(elasticJob)
	needSync := r.satisfiedExpections(elasticJob)
	if !needSync {
		log.Info("reconcile skipped, job does not need to sync")
		return ctrl.Result{}, nil
	}

	if elasticJob.DeletionTimestamp != nil {
		log.Info("reconcile skipped, job has been deleted.")
		return ctrl.Result{}, nil
	}

	// Set default priorities for elastic job
	scheme.Scheme.Default(elasticJob)

	// Set default cleanPodPolicy for job
	if elasticJob.Spec.RunPolicy.CleanPodPolicy == nil {
		elasticJob.Spec.RunPolicy.CleanPodPolicy = &defaultCleanPodPolicy
	}

	// Use common to reconcile the job related pod and service
	err = r.jobController.ReconcileJobs(elasticJob, elasticJob.Spec.ReplicaSpecs, elasticJob.Status.JobStatus, &elasticJob.Spec.RunPolicy)
	if err != nil {
		log.Infof("Reconcile ElasticJob error %v", err)
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (r *ElasticJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// setup FieldIndexer to inform the manager that this controller owns pods and services,
	// so that it will automatically call Reconcile on the underlying ElasticJob when a Pod or Service changes, is deleted, etc.
	if err := mgr.GetFieldIndexer().IndexField(&corev1.Pod{}, jobOwnerKey, func(rawObj runtime.Object) []string {
		pod := rawObj.(*corev1.Pod)
		owner := metav1.GetControllerOf(pod)
		if owner == nil {
			return nil
		}

		// Make sure owner is ElasticJob Controller.
		if owner.APIVersion != r.GetAPIGroupVersion().Version || owner.Kind != r.GetAPIGroupVersionKind().Kind {
			return nil
		}

		return []string{owner.Name}
	}); err != nil {
		return err
	}

	if err := mgr.GetFieldIndexer().IndexField(&corev1.Service{}, jobOwnerKey, func(rawObj runtime.Object) []string {
		svc := rawObj.(*corev1.Service)
		owner := metav1.GetControllerOf(svc)
		if owner == nil {
			return nil
		}

		if owner.APIVersion != r.GetAPIGroupVersion().Version || owner.Kind != r.GetAPIGroupVersionKind().Kind {
			return nil
		}

		return []string{owner.Name}
	}); err != nil {
		return err
	}

	// Setup ElasticJobReconciler
	r.Client = mgr.GetClient()
	r.Scheme = mgr.GetScheme()

	// Create k8s clients to list pods and service objects
	kubeClientSet := kubernetes.NewForConfigOrDie(mgr.GetConfig())

	r.jobController = common.JobController{
		Controller:    r,
		Config:        common.JobControllerConfiguration{EnableGangScheduling: false},
		Expectations:  k8scontroller.NewControllerExpectations(),
		WorkQueue:     workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), r.ControllerName()),
		Recorder:      mgr.GetEventRecorderFor(r.ControllerName()),
		KubeClientSet: kubeClientSet,
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.ElasticJob{}).
		Owns(&corev1.Pod{}).
		Owns(&corev1.Service{}).
		WithEventFilter(predicate.Funcs{CreateFunc: onDependentCreateFunc(r), DeleteFunc: onDependentDeleteFunc(r)}).
		Complete(r)
}

func (r *ElasticJobReconciler) ControllerName() string {
	return controllerName
}

func (r *ElasticJobReconciler) GetAPIGroupVersionKind() schema.GroupVersionKind {
	return v1alpha1.GroupVersion.WithKind(v1alpha1.Kind)
}

func (r *ElasticJobReconciler) GetAPIGroupVersion() schema.GroupVersion {
	return v1alpha1.GroupVersion
}

func (r *ElasticJobReconciler) GetGroupNameLabelValue() string {
	return v1alpha1.GroupVersion.Group
}

func (r *ElasticJobReconciler) GetDefaultContainerName() string {
	return v1alpha1.DefaultContainerName
}

func (r *ElasticJobReconciler) GetDefaultContainerPortNumber() int32 {
	// elastic job doesn't use fixed port
	return -1
}

func (r *ElasticJobReconciler) GetDefaultContainerPortName() string {
	// elastic job doesn't use fixed port
	return ""
}

func (r *ElasticJobReconciler) GetJobRoleKey() string {
	return elasticJobRoleLabel
}

func (r *ElasticJobReconciler) IsMasterRole(replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec,
	rtype commonv1.ReplicaType, index int) bool {
	// This method is from commonv1.ControllerInterface.
	// All workers in PET are equivalent and this doesn't apply to PET, so always return false here.
	return false
}

// SetClusterSpec sets the cluster spec for the pod
func (r *ElasticJobReconciler) SetClusterSpec(job interface{}, podTemplate *corev1.PodTemplateSpec, rtype, index string) error {
	return SetClusterSpecForPod(job, podTemplate)
}
