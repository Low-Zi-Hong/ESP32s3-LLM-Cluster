# ESP32s3-LLM-Cluster

A distributed pipeline inference engine on multiple ESP32S3 running 1.58-bit (BitNet) Language model. 

## Architecture

This project runs a sliced 0.5B LLM across a cluster of 7 ESP32s3. One act as master and others are node. The master node runs the tokenizer and embeding and the other attention layer and MLP ran on the nodes. The master and node communicate through high speed SPI Daisy-Chain.

## Getting Started

pls refer [[workflow]] to start with the project.


# Benchmark


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)