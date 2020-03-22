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
	v1 "github.com/kubeflow/common/pkg/apis/common/v1"
	commonutil "github.com/kubeflow/common/pkg/util"
	logger "github.com/kubeflow/common/pkg/util"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"reflect"
)

// Reasons for job events.
const (
	FailedDeleteJobReason     = "FailedDeleteJob"
	SuccessfulDeleteJobReason = "SuccessfulDeleteJob"

	// ElasticJobCreatedReason is added in a job when it is created.
	ElasticJobCreatedReason    = "ElasticJobCreated"
	ElasticJobSucceededReason  = "ElasticJobSucceeded"
	ElasticJobRunningReason    = "ElasticJobRunning"
	ElasticJobFailedReason     = "ElasticJobFailed"
	ElasticJobRestartingReason = "ElasticJobRestarting"
)

// GetJobFromInformerCache returns the Job from Informer Cache
func (r *ElasticJobReconciler) GetJobFromInformerCache(namespace, name string) (metav1.Object, error) {
	job := &v1alpha1.ElasticJob{}
	log := logger.LoggerForJob(job)

	// Default reader for ElasticJob is cache reader.
	err := r.Get(context.Background(), types.NamespacedName{Namespace: namespace, Name: name}, job)
	if err != nil {
		if errors.IsNotFound(err) {
			log.Errorf("ElasticJob %s/%s not found. %v", namespace, name, err)
		} else {
			log.Error(err, "failed to get job %s/%s from informer cache. %v", namespace, name, err)
		}
		return nil, err
	}
	return job, nil
}

// GetJobFromAPIClient returns the Job from API server
func (r *ElasticJobReconciler) GetJobFromAPIClient(namespace, name string) (metav1.Object, error) {
	job := &v1alpha1.ElasticJob{}
	log := logger.LoggerForJob(job)

	clientReader, err := getClientReaderFromClient(r.Client)
	if err != nil {
		return nil, err
	}
	err = clientReader.Get(context.Background(), types.NamespacedName{Namespace: namespace, Name: name}, job)
	if err != nil {
		if errors.IsNotFound(err) {
			log.Errorf("ElasticJob %s/%s not found. %v", namespace, name, err)
		} else {
			log.Errorf("failed to get job %s/%s from api-server. %v", namespace, name, err)
		}
		return nil, err
	}
	return job, nil
}

// DeleteJob deletes the job
func (r *ElasticJobReconciler) DeleteJob(job interface{}) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)
	if err := r.Delete(context.Background(), elasticJob); err != nil {
		r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeWarning, FailedDeleteJobReason, "Error deleting: %v", err)
		log.Errorf("failed to delete job %s/%s, %v", elasticJob.Namespace, elasticJob.Name, err)
		return err
	}
	r.jobController.Recorder.Eventf(elasticJob, corev1.EventTypeNormal, SuccessfulDeleteJobReason, "Deleted job: %v", elasticJob.Name)
	log.Infof("job %s/%s has been deleted", elasticJob.Namespace, elasticJob.Name)
	return nil
}

// UpdateJobStatus updates the job status and job conditions
func (r *ElasticJobReconciler) UpdateJobStatus(job interface{}, replicas map[v1.ReplicaType]*v1.ReplicaSpec, jobStatus *v1.JobStatus) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	log := logger.LoggerForJob(elasticJob)

	for rtype, spec := range replicas {
		status := jobStatus.ReplicaStatuses[rtype]

		succeeded := status.Succeeded
		expected := *(spec.Replicas) - succeeded
		running := status.Active
		failed := status.Failed

		log.Infof("ElasticJob=%s, ReplicaType=%s expected=%d, running=%d, succeeded=%d , failed=%d",
			elasticJob.Name, rtype, expected, running, succeeded, failed)

		if rtype == v1.ReplicaType(v1alpha1.ElasticReplicaTypeWorker) {
			if running > 0 {
				msg := fmt.Sprintf("ElasticJob %s is running.", elasticJob.Name)
				err := commonutil.UpdateJobConditions(jobStatus, v1.JobRunning, ElasticJobRunningReason, msg)
				if err != nil {
					log.Errorf("Append job condition error: %v", err)
					return err
				}
			}
			// when all workers succeed, the job is finished.
			if expected == 0 {
				msg := fmt.Sprintf("ElasticJob %s is successfully completed.", elasticJob.Name)
				log.Info(msg)
				r.jobController.Recorder.Event(elasticJob, corev1.EventTypeNormal, ElasticJobSucceededReason, msg)
				if jobStatus.CompletionTime == nil {
					now := metav1.Now()
					jobStatus.CompletionTime = &now
				}
				err := commonutil.UpdateJobConditions(jobStatus, v1.JobSucceeded, ElasticJobSucceededReason, msg)
				if err != nil {
					log.Errorf("Append job condition error: %v", err)
					return err
				}
				return nil
			}
		}
		if failed > 0 {
			if spec.RestartPolicy == v1.RestartPolicyExitCode {
				msg := fmt.Sprintf("ElasticJob %s is restarting because %d %s replica(s) failed.", elasticJob.Name, failed, rtype)
				r.jobController.Recorder.Event(elasticJob, corev1.EventTypeWarning, ElasticJobRestartingReason, msg)
				err := commonutil.UpdateJobConditions(jobStatus, v1.JobRestarting, ElasticJobRestartingReason, msg)
				if err != nil {
					log.Errorf("Append job condition error: %v", err)
					return err
				}
			} else {
				msg := fmt.Sprintf("ElasticJob %s is failed because %d %s replica(s) failed.", elasticJob.Name, failed, rtype)
				r.jobController.Recorder.Event(elasticJob, corev1.EventTypeNormal, ElasticJobFailedReason, msg)
				if elasticJob.Status.CompletionTime == nil {
					now := metav1.Now()
					elasticJob.Status.CompletionTime = &now
				}
				err := commonutil.UpdateJobConditions(jobStatus, v1.JobFailed, ElasticJobFailedReason, msg)
				if err != nil {
					log.Errorf("Append job condition error: %v", err)
					return err
				}
			}
		}
	}

	// Some workers are still running, leave a running condition.
	msg := fmt.Sprintf("ElasticJob %s is running.", elasticJob.Name)
	log.Infof(msg)

	if err := commonutil.UpdateJobConditions(jobStatus, v1.JobRunning, ElasticJobRunningReason, msg); err != nil {
		log.Errorf("failed to update ElasticJob conditions %v", err)
		return err
	}

	return nil
}

// UpdateJobStatusInApiServer updates the job status in to cluster.
func (r *ElasticJobReconciler) UpdateJobStatusInApiServer(job interface{}, jobStatus *v1.JobStatus) error {
	elasticJob, ok := job.(*v1alpha1.ElasticJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of ElasticJob", elasticJob)
	}

	// Job status passed in differs with status in job, update in basis of the passed in one.
	if !reflect.DeepEqual(&elasticJob.Status.JobStatus, jobStatus) {
		elasticJob = elasticJob.DeepCopy()
		elasticJob.Status.JobStatus = *jobStatus.DeepCopy()
	}

	result := r.Update(context.Background(), elasticJob)
	if result != nil {
		logger.LoggerForJob(elasticJob).Error(result, " failed to update ElasticJob conditions in the API server")
		return result
	}

	return nil
}
