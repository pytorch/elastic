#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc

import boto3


class AwsSessionProvider:
    """
    Provides AWS credentials in the form of boto3 Session.
    This class may be sub-classed to provide custom methods
    of getting aws_access_key_id and aws_secret_access_key.
    Child classes are expected to provide overriding implementations
    of the three `_get_*` methods below.

    When used directly, it follows the default credential
    lookup chain as documented in:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    """

    def get_session(self, region=None):
        access_key = self._get_access_key()
        secret_key = self._get_secret_key()
        session_token = self._get_session_token()

        # either both access and secret keys are None
        # or both are not None; just check one to assume
        # the presence of the other
        if access_key is None:
            return boto3.session.Session()
        else:
            return boto3.session.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                region_name=region,
            )

    def _get_access_key(self):
        """
        Returns the aws_access_key_id. Override when sub-classing.
        """
        return None

    def _get_secret_key(self):
        """
        Returns the aws_secret_access_key. Override when sub-classing.
        """
        return None

    def _get_session_token(self):
        """
        Returns the aws_session_token. Override when sub-classing.
        """
        return None
