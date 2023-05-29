# Docker ContainerをGitHubにつなげる
## gitのインストール
```bash
sudo apt-get update && sudo apt-get install -y git
```
確認
```bash
git version
# git version 2.25.1
```

## sshの公開鍵・秘密鍵の作成
ディレクトリ`.ssh`を確認 > 無ければ作成
```bash
mkdir ~/.ssh
ssh-keygen -t rsa
```

## GitHubに公開鍵をアップ
公開鍵の取得
```bash
cat ~/.ssh/id_rsa.pub
```
表示されている文章を**全てコピー**

[GitHubの設定](https://github.com/settings/ssh)にアップ

## gitコマンドでプッシュ
```bash
git init
git remote add [github@...]
git add --all
git commit -m "first commit"
git branch -M main
git push origin main
```
