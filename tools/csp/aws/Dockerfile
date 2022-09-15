FROM nvcr.io/ea-bignlp/bignlp-training:22.08.01-py3

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=1.4.0-aws

# EFA
RUN apt-get update && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer/DEBS/UBUNTU2004/$(uname -m) && \
    dpkg -i libfabric*.deb && \
    dpkg -i efa-profile*.deb && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/* && \
    /opt/amazon/efa/bin/fi_info --version

ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:$PATH

# NCCL EFA Plugin
RUN mkdir -p /tmp && \
    cd /tmp && \
    curl -LO https://github.com/aws/aws-ofi-nccl/archive/refs/tags/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    tar -xzf /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    rm /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    mv aws-ofi-nccl-${AWS_OFI_NCCL_VERSION} aws-ofi-nccl && \
    cd /tmp/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --with-mpi=/usr/local/mpi && \
    make -j$(nproc) install && \
    rm -rf /tmp/aws-ofi/nccl

# NCCL
RUN echo "/usr/local/lib"      >> /etc/ld.so.conf.d/local.conf && \
    echo "/opt/amazon/efa/lib" >> /etc/ld.so.conf.d/efa.conf &&  \
    ldconfig

ENV OMPI_MCA_pml=^ucx                \
    OMPI_MCA_btl=^openib,uct         \
    NCCL_TOPO_FILE=/usr/local/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml

ENV RDMAV_FORK_SAFE=1             \
    FI_PROVIDER=efa               \
    FI_EFA_USE_DEVICE_RDMA=1

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
