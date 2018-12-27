#!/bin/bash

bq load --source_format NEWLINE_DELIMITED_JSON jokes.reddit reddit.json id,title,body,score:INTEGER
