/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

package v1alpha1

const (
	// Kind is the kind name.
	Kind = "ElasticJob"

	DefaultContainerName     = "elasticjob-worker"
	DefaultContainerPortName = "elasticjob-port"
	DefaultPort              = 10291
)
