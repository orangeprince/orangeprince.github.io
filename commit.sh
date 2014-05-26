#!/usr/bin/env bash
jekyll build -s . -d ../www/
rsync -r ../www/* root@orangeprince.info:/var/www
