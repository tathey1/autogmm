# hash:sha256:efc65f36f2162d9b0138f07c96a878bef66050f3cfe3dfaeb85ea3969e73670b
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        libgit2-dev=0.28.4+dfsg.1-2 \
        libssl-dev=1.1.1f-1ubuntu2.4 \
        pkg-config=0.29.1-0ubuntu4 \
        r-base=3.6.3-2 \
        r-base-dev=3.6.3-2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    colorcet==2.0.6 \
    graspologic==0.2.0 \
    pyreadr==0.4.2

RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && echo 'options(Ncpus = parallel::detectCores())' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && Rscript -e 'options(warn=2); install.packages("remotes")'
RUN Rscript -e 'remotes::install_version("mclust", "5.4.7")'

COPY postInstall /
RUN /postInstall
