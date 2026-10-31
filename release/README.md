# 预编译引擎 / Prebuilt engine

`strata-0.1.54-sm60-linux-x86_64.tar.gz` (243 MB) 是本仓库作者在下列环境编译出的引擎二进制：

| | |
|---|---|
| Strata | v0.2.8 源码，engine 0.1.54 |
| CUDA | 12.9.86 |
| 编译器 | gcc-13（宿主编译器）+ nvcc 12.9 |
| 目标架构 | `60;61;70;86`（P100 / V100 / RTX 30 系） |
| 构建参数 | `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON` |

```
sha256  b2f4ff55b5a76d042bbcb7d04e167768f5ae55f9095087c546704d86af08bbd0
size    243 MB (tar.gz), 444 MB 解压后
```

## ⚠️ 不要用 `git add` 提交它

GitHub 单文件上限 **100 MB**，这个包 243 MB，提交会被拒绝。**必须走 Release 附件。**

## 发布方式 A：网页上传（最简单）

1. 在 GitHub 上建好空仓库（不要勾 README / .gitignore / license）
2. 把本目录**除 `release/` 之外**的文件推上去
3. 仓库页 → **Releases → Draft a new release**
4. Tag 填 `v0.1.54-sm60`，标题随意
5. 把 `release/strata-0.1.54-sm60-linux-x86_64.tar.gz` 拖进附件框
6. 正文里粘贴上面的 sha256 和构建设置 → **Publish release**

## 发布方式 B：命令行

```bash
# 装 git 和 gh 之后
cd p100-strata-opt
git init -b main && git add -A && git commit -m "Pascal (P100) support for Strata + measured tuning"
git remote add origin https://github.com/<你>/<仓库>.git
git push -u origin main

gh release create v0.1.54-sm60 release/strata-0.1.54-sm60-linux-x86_64.tar.gz \
   --title "Strata engine 0.1.54 for Tesla P100 (sm_60)" \
   --notes "Built from v0.2.8 with CUDA 12.9 / gcc-13, archs 60;61;70;86. sha256 b2f4ff55b5a76d042bbcb7d04e167768f5ae55f9095087c546704d86af08bbd0"
```

## ⚠️ 更重要的一点：这个包不能配 `strata launch` 用

`strata launch` 会拿签名 feed 校验引擎，发现二进制被换过就**重新下载官方包覆盖**。
必须用：

```bash
strata serve --profile <你的 profile>.json --host 0.0.0.0 --port 8080
```

具体替换步骤见包内的 `INSTALL.md`。
