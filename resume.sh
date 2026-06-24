#!/bin/bash
# Resume sweep from (var=5.0, cov=0.25), then var=7.0 and var=10.0.
# Completed: var=2.0 (all cov), var=5.0 (cov=0.04..0.15)

ROUND=resume bash sweep.sh
