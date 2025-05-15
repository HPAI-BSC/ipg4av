Here are some paper pages that you can use as reference as to what to add to your paper website. Check what may align
with your paper.

- https://alfworld.github.io
- https://askforalfred.github.io
- https://hypernerf.github.io
- https://nerfies.github.io
- https://language-to-reward.github.io
- https://vimalabs.github.io
- https://eureka-research.github.io


### Installing Ruby and Jekyll

```shell
sudo apt install -y ruby ruby-dev 
sudo gem install jekyll bundler
sudo bundle install
```

### Using the template

1. Edit `_config.yml`
   - Set the `description` attribute to the name of the paper. This makes it appear as the webpage title in your browser.
2. Edit `index.md`
   - Replace the `feature_image` for a background image of your choice.
   - Edit the `feature_text` attribute at the start, replacing `title` and `authorN` by the corresponding data in your paper.
   - Edit the Jekyll templates (and add more if needed) to show the link buttons you need to show.
   - Populate the file with the content that will appear in your page, using Markdown. Be sure that at least an abstract and "cite as" sections are present.

### Testing in your local PC

Create a local server like this:

```shell
bundle exec jekyll serve
```

Then access http://127.0.0.1:4000/ to see the frontpage of your page.

### Uploading to Github

To be documented