/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package controllers

import (
	"errors"
	"fmt"
	"github.com/kubeflow/common/pkg/apis/common/v1"
	"github.com/pytorch/elastic/kubernetes/api/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// computeDesiredReplicas retrieve user's replica setting in specs
func computeDesiredReplicas(elasticJob *v1alpha1.ElasticJob) (int32, error) {
	workerSpecs, exist := elasticJob.Spec.ReplicaSpecs[v1.ReplicaType(v1alpha1.ElasticReplicaTypeWorker)]
	if !exist {
		return 0, fmt.Errorf("elasticJob %v doesn't have %s", elasticJob, v1alpha1.ElasticReplicaTypeWorker)
	}

	return *workerSpecs.Replicas, nil
}

func getClientReaderFromClient(client client.Client) (client.Reader, error) {
	if dr, err := getDelegatingReader(client); err != nil {
		return nil, err
	} else {
		return dr.ClientReader, nil
	}
}

// getDelegatingReader try to extract DelegatingReader from client.
func getDelegatingReader(c client.Client) (*client.DelegatingReader, error) {
	dc, ok := c.(*client.DelegatingClient)
	if !ok {
		return nil, errors.New("cannot convert from Client to DelegatingClient")
	}
	dr, ok := dc.Reader.(*client.DelegatingReader)
	if !ok {
		return nil, errors.New("cannot convert from DelegatingClient.Reader to Delegating Reader")
	}
	return dr, nil
}
