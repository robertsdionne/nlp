#!/bin/bash

for test in $(ls *Test.lua) tests.lua
do
  luajit $test
done
