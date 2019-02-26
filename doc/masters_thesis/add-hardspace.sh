#!/bin/bash

find rozdzialy -name "*.tex" -type f -print0 | xargs -0 sed -i 's/ i / i~/g;s/ a / a~/g;s/ u / u~/g;s/ z / z~/g;s/ na / na~/g;s/ w / w~/g;s/ Z / Z~/g;s/ W / W~/g;s/ o / o~/g'
