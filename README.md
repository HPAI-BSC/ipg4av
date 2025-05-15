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
   - Set the `baseurl` to the name of your final repository in Github. You need to leave the `/` before your repo name.
   - Set the `feature_image` to something else.
   - Set the `description` attribute to the name of the paper. This makes it appear as the webpage title in your browser.
2. Edit `index.md`
   - Replace the `feature_image` for a background image of your choice.
   - Edit the `feature_text` attribute at the start, replacing `title` and `authorN` by the corresponding data in your paper and editing the affiliations and institution names as needed.
   - Edit the Jekyll templates (and add more if needed) to show the link buttons you need to show.
   - Populate the file with the content that will appear in your page, using Markdown. Be sure that at least an abstract and "cite as" sections are present.

### Testing in your local PC

Create a local server like this:

```shell
bundle exec jekyll serve
```

Then access http://127.0.0.1:4000/ to see the frontpage of your page.

### Uploading to Github

1. Create a new **public** Github repository for your webpage. Name it as you want.
2. Clone your new repository in your local PC: `git clone git@github.com:HPAI-BSC/<your-repo>.git`.
3. Copy the contents of the template (except for the `README.md`) into your paper's repo.
4. Check that the template's webpage (http://127.0.0.1:4000/) is displayed correctly by using `bundle exec jekyll serve`.
5. Commit and push the template's contents into the repo.
6. In Github, go to **Settings > Pages > Branch** and select the `main` branch and the `/docs` folder as the sources of your webpage.
7. Populate your webpage following the steps in the "_Using the template_" section.
8. Commit and push your content. Your webpage will be at https://hpai-bsc.github.io/your-project. 