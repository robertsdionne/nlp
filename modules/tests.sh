#!/bin/bash

for test in $(ls *Test.lua) tests.lua
do
  th $test && echo $test' PASSED' || echo $test' FAILED'
done
