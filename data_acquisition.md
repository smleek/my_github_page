# Utah Kindergarten Immunization Data

## Introduction
Public health experts throughout the country are concerned about the rise in vaccine hesitancy. The threshold for herd immunity is fairly high at 95%, so after accounting for those who are medically unable to receive vaccinations (infants, immunocompromised, etc), the space for vaccine-averse people in an ideally immune society is really quite limited. 

Utah is an interesting case study, because despite being very conservative throughout, Utah tends to rank fairly high for educational attainment. I was curious whether this high level of education would contribute to high vaccine rates, or if conservative patterns would reign. So, using publically available data regarding kindergarten immunizations, I sought to gain a little bit of insight into vaccine rates. 

## The Process

### Data Scraping and Collection 
The data regarding kindergarten immunizations is available via an RShiny app hosted on url. I scraped this data using this. 

I joined this dataset with an elementary school directory available at this url. 

Since these were the only two datasets I used for this analysis, all the data here is free to be shared. 

### Data Cleaning
If you're curious as to how I managed to curate my dataset, I'll explain a bit more here. 

The first and most time-consuming step was to scrape the vaccine exemption data from [text](https://avrpublic.dhhs.utah.gov/imms_dashboard/). I found that several Python packages were required to make this step possible, as I am inexperienced with web scraping. Selenium, BeautifulSoup4, and StringIO were the most integral to the effort. One thing to keep in mind for this data was that the data was available page-by-page, requiring the script to click "Next" about 60 times to get every possible data point. I eventually procured what I called vax_exemptions: 

!['Preview of vax_exemptions, curated by Sujin'](images/vax_exemptionshead.png)

The next step was to get a directory of all schools in Utah, so that I could group schools with different variables. [text](https://schools.utah.gov/schoolsdirectory) allows you to filter schools to your liking and download a .csv file, so that was very helpful. I ended up dropping several variables from this data frame, and then did a pandas inner join to continue with my analysis. Disappointingly, the two data frames were missing respective schools due to different criteria and data collection methods, but as someone not very knowledgeable about Utah schools, I decided to continue with my analysis. 

### Variable dictionary
Here is a short preview of the data: 

!['Preview of Utah Kindergarten Immunization Dataset, curated by Sujin'](images/imms_head.png)

School: Facility name. An interesting side-finding from this analysis was that many elementary schools in Utah share the same name. 
City: the city in which the school is located
Adequately immunized %: the percentage of students who have completed all required vaccinations, without exceptions. 
Conditionally enrolled %: the percentage of students who are conditionally enrolled, having completed only some of the vaccinations, on the condition that they work to receive their further required vaccinations. 
Extended conditionally enrolled %: at the end of the conditionally enrolled period, administrators can choose to extend that period for students. 
Exempt %: The percentage of students who are exempt from completing all required vaccinations, whether for medical or personal reasons. 
District/LEA: The district which the school belongs to
Grades: The grade levels that the school serves
Address: The street address of the school 
Zip: The zip code for the address of the school 
Map: the latitude-longitude coordinates for the school. 
school_type: A column denoting whether the school is private, public, or chartered. 

You'll notice that I left a lot of location-based variables in that were not used; I had hoped to explain data via location clustering, but could not cluster the data into anything useful. 

## My Analysis 

My first takeaway as I scrolled through the .csv file was that only a few schools in Utah are past the 95% threshold. The Department of Health and Human Services reports only 86.9% of kindergartners statewide to be adequately immunized, and most schools follow that trend. However, as I scrolled through the data, I often found that when I saw a school with an alarmingly low or surprisingly high vaccination rate, it was a private or charter school. Examples are the Walden School of Liberal Arts at 49% and Saint Joseph Catholic Elementary School at 100%. 

I formed the hypothesis that perhaps public schools had lower variance in vaccination rates than private and charter schools, and that the outliers seen in this histogram were likely mostly private or charter schools. 

!['Histogram of Overall Adequate Immunization Percentages in Kindergarteners'](images/hist.png)

I decided to test this hypothesis by first visualizing the data through boxplots. 

!['Boxplot Comparing Percent Adequately Immunized Between Utah Public Schools and Utah Non-Traditional Private or Charter Schools'](images/imms_boxplot.png)

My hypothesis was already looking somewhat correct, so I decided to visualize further with a violin plot, which confirmed my opinions. Violin plots are useful for seeing more clearly the distributions of data. 

!['Violin Plot Comparing Percent Adequately Immunized Between Utah Public Schools and Utah Non-Traditional Private or Charter Schools'](images/imms_boxplot.png)

A quick calculation of variances showed that the variance of public schools was equal to approximately 57, while the variance of private or charter schools was equal to approximately 210. This is already quite conclusive, but I decided to test the differences in variance with a measurable Levene test; while a t-test would have done the same thing, this data is not normally distributed and a Levene test is more robust against non-normal distributions. 

My Levene test revealed that the variances were very different, with a very low p-value of 4.744105060651353e-06. 

## Conclusions

I conclude that there is a difference in variance of vaccination rates between public and charter/private schools, with public schools having lower variance. As human instinct would tell you, families choosing to attend private or charter schools will come from a more varied distribution than families choosing to attend their default public school. 

## Links and Additional Information 

A link to my code: [text](https://github.com/smleek/data_acquisition_blog)

An article on herd immunity and measles: [text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12581858/)