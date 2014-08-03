#!/bin/bash

build/test/test_all.testbin --gtest_filter="*$1*" 2>&1 | tee $1"TestOutput"
