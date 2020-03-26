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
	logger "github.com/kubeflow/common/pkg/util"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"strconv"
)

// CreatePod creates the pod of the job
func (r *ElasticJobReconciler) CreatePod(job interface{}, pod *corev1.Pod) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)
	log.Infof("Creating pod %s/%s, Job name: %s.", pod.Namespace, pod.Name, elasticJob.GetName())

	if err := r.Create(context.Background(), pod); err != nil {
		log.Infof("Error building a pod via Elastic operator: %s", err.Error())
		return err
	}

	return nil
}

// DeletePod deletes the pod of the job
func (r *ElasticJobReconciler) DeletePod(job interface{}, pod *corev1.Pod) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of TorchElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)
	log.Infof("Deleting pod %s/%s, Job name: %s", pod.Namespace, pod.Name, elasticJob.GetName())
	if err := r.Delete(context.Background(), pod); err != nil {
		r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeWarning, common.FailedDeletePodReason, "Error deleting: %v", err)
		return err
	}

	r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeNormal, common.SuccessfulDeletePodReason, "Deleted pod: %v", pod.Name)

	return nil
}

// GetPodsForJob returns the pods managed by the job. This can be achieved by selecting pods using label key "job-name"
// i.e. all pods created by the job will come with label "job-name" = <this_job_name>
func (r *ElasticJobReconciler) GetPodsForJob(obj interface{}) ([]*corev1.Pod, error) {
	job, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	// List all pods to include those that don't match the selector anymore
	// but have a ControllerRef pointing to this controller.
	podlist := &corev1.PodList{}
	if err := r.List(context.Background(), podlist, client.InNamespace(job.GetNamespace()),
		client.MatchingLabels(r.jobController.GenLabels(job.GetName()))); err != nil {
		return nil, err
	}

	return convertPodList(podlist.Items), nil
}

// convertPodList convert pod list to pod pointer list
func convertPodList(list []corev1.Pod) []*corev1.Pod {
	if list == nil {
		return nil
	}
	ret := make([]*corev1.Pod, 0, len(list))
	for i := range list {
		ret = append(ret, &list[i])
	}
	return ret
}

// Set pod environment set for ElasticJob
func SetClusterSpecForPod(job interface{}, podTemplate *corev1.PodTemplateSpec) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	desiredReplicas, err := computeDesiredReplicas(elasticJob)
	if err != nil {
		return err
	}

	// Set default value if minReplicas and maxReplicas are not set
	var minReplicas, maxReplicas int32
	if elasticJob.Spec.MinReplicas != nil {
		minReplicas = *elasticJob.Spec.MinReplicas
	} else {
		minReplicas = desiredReplicas
	}

	if elasticJob.Spec.MaxReplicas != nil {
		maxReplicas = *elasticJob.Spec.MaxReplicas
	} else {
		maxReplicas = desiredReplicas
	}

	for i := range podTemplate.Spec.Containers {
		if len(podTemplate.Spec.Containers[i].Env) == 0 {
			podTemplate.Spec.Containers[i].Env = make([]corev1.EnvVar, 0)
		}

		podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
			Name:  "RDZV_ENDPOINT",
			Value: elasticJob.Spec.RdzvEndpoint,
		})
		podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
			Name:  "JOB_ID",
			Value: elasticJob.Name,
		})
		podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
			Name:  "SIZE",
			Value: strconv.Itoa(int(desiredReplicas)),
		})
		podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
			Name:  "MIN_SIZE",
			Value: strconv.Itoa(int(minReplicas)),
		})
		podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
			Name:  "MAX_SIZE",
			Value: strconv.Itoa(int(maxReplicas)),
		})
	}

	return nil
}
