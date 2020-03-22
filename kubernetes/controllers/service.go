/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package controllers

import (
	"context"
	"fmt"
	"github.com/kubeflow/common/pkg/controller.v1/common"
	commonutil "github.com/kubeflow/common/pkg/util"
	logger "github.com/kubeflow/common/pkg/util"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// CreateService creates the service
func (r *ElasticJobReconciler) CreateService(job interface{}, service *corev1.Service) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)
	log.Infof("Creating service %s/%s, Job name: %s ", service.Namespace, service.Name, elasticJob.GetName())

	if err := r.Create(context.Background(), service); err != nil {
		log.Infof("Create service %s/%s error %s", service.Namespace, service.Name, err)
	}

	return nil
}

// DeleteService deletes the service
func (r *ElasticJobReconciler) DeleteService(job interface{}, name string, namespace string) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)
	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}}
	log.Infof("Deleting service %s/%s, Job name %s", service.Namespace, service.Name, elasticJob.GetName())

	if err := r.Delete(context.Background(), service); err != nil {
		if commonutil.IsSucceeded(elasticJob.Status.JobStatus) {
			//r.recorder.Eventf(elasticJob, corev1.EventTypeNormal, job_controller.SuccessfulDeleteServiceReason, "Deleted service: %v", name)
			return nil
		}

		r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeWarning, common.FailedDeleteServiceReason, "Error deleting: %v", err)
		return fmt.Errorf("unable to delete service: %v", err)
	}

	r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeNormal, common.SuccessfulDeleteServiceReason, "Deleted service: %v", name)
	return nil
}

// GetServicesForJob returns the services managed by the job. This can be achieved by selecting services using label key "job-name"
// i.e. all services created by the job will come with label "job-name" = <this_job_name>
func (r *ElasticJobReconciler) GetServicesForJob(obj interface{}) ([]*corev1.Service, error) {
	job, err := meta.Accessor(obj)
	if err != nil {
		return nil, fmt.Errorf("%+v is not a type of TorchElasticJob", job)
	}
	// List all pods to include those that don't match the selector anymore
	// but have a ControllerRef pointing to this controller.
	serviceList := &corev1.ServiceList{}
	if err := r.List(context.Background(), serviceList, client.InNamespace(job.GetNamespace()),
		client.MatchingLabels(r.jobController.GenLabels(job.GetName()))); err != nil {
		return nil, err
	}

	ret := convertServiceList(serviceList.Items)

	return ret, nil
}

// convertServiceList convert service list to service point list
func convertServiceList(list []corev1.Service) []*corev1.Service {
	if list == nil {
		return nil
	}
	ret := make([]*corev1.Service, 0, len(list))
	for i := range list {
		ret = append(ret, &list[i])
	}
	return ret
}
