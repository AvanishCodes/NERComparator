class Data():
	def __init__(self, texts: list=[]):
		self.texts = texts
		# self.texts = [
		# 	'Management Board Advisers Integra Micro Systems',
		# 	'Integra Micro Systems',
		# 	'Home',
		# 	'About Us',
		# 	'Why Integra?',
		# 	'Leadership',
		# 	'Affiliated Companies',
		# 	'Partners Customers',
		# 	'CSR',
		# 	'Contact Us',
		# 	'Domains',
		# 	'Banking Payment Systems',
		# 	'Agent Assisted Services',
		# 	'eGovernance',
		# 	'Biometric Enabler Systems',
		# 	'Products',
		# 	'Financial Inclusion',
		# 	'Multifactor authentication',
		# 	'Business Applications',
		# 	'Imaging Software',
		# 	'Solutions',
		# 	'Branchless Banking',
		# 	'Mobile Payments',
		# 	'Tab Banking',
		# 	'Closed user group wallet systems',
		# 	'Loan Maintenance',
		# 	'IPMS',
		# 	'Video Surveillance',
		# 	'ICR/OCR/OMR',
		# 	'LoanBOT',
		# 	'ChatBOT',
		# 	'Distribution',
		# 	'Document Imaging',
		# 	'Open Source Software',
		# 	'Biometric Solutions',
		# 	'Disaster Recovery',
		# 	'LogicalDOC',
		# 	'Resources',
		# 	'News And Opinions',
		# 	'Downloads',
		# 	'Careers',
		# 	'Leadership',
		# 	'Provider hitechnology products solutions for the Government BFSI Telecom space',
		# 	'About us > Leadership',
		# 	'Management',
		# 	'Board Advisors',
		# 	'Mahesh Kumar Jain Chairman CEO Managing Director',
		# 	"Mahesh is the cofounder Chairman the Integra group companies. He has led Integra's efforts in building award winning products services in multiple market segments ranging from banking imaging mobile technologies education financial inclusion.",
		# 	'He is currently focused on building technologies to bridge the accessibility gap with several initiatives in financial inclusion eGovernance. He has also been associated with education carbon green plantation societies for many years is the cofounder a few organizations in this space.',
		# 	'Vimla Jogesh Head Value Added Distribution',
		# 	'Vimla heads the value added distribution business at Integra. She leads the setup management relationships with several OEMS nurtures the partner ecosystem across the country. She leads teams to achieve sales operational aspirations the business unit.',
		# 	'Vimla brings over thirty years rich experience in building alliances sales marketing operations has held several leadership positions in the Government private sectors.',
		# 	'Ashwin Shankar Head EGovernance Solutions',
		# 	'Ashwin brings over two decades experience in building solutions from the ground up in eGovernance banking financial services. He has been driving the egovernance business Integra has executed several projects in India as well as in eight African countries.',
		# 	'Prior to joining Integra Ashwin held management positions at HSBC in Credit Personal Private Banking where he was responsible for introducing new products in several countries.',
		# 	'Manjunatha HS Head iMFAST Operations',
		# 	'Manjunatha oversees all revenue customer engagements daytoday operations the financial inclusion business at Integra. He shoulders responsibility in business leadership procurement supply chain agent assisted commerce sales marketing activities related to this business unit.',
		# 	'Manjunatha brings over sixteen years experience in sales marketing business development for Integra’s products third party products including open source software imaging communications.',
		# 	'K Sivakkumar Head Engineering',
		# 	'Siva is an experienced passionate professional with over two decades experience in engineering technology with the IT services industry. He heads the RD financial inclusion services at Integra.',
		# 	'Experienced in infrastructure networking information security systems integration implementation services Siva has been instrumental in creating the iMFAST platform putting in place robust solutions to meet the fast changing challenging market conditions.',
		# 	'Bhaskar Jyoti Phukan Head New Initiatives',
		# 	'Bhaskar brings over two decades years experience in education quality project management activities. He has been instrumental in assisting the organization in all new initiatives building alliances contribution in policy making.',
		# 	'Prior to joining Integra Bhaskar held teaching positions in Physics for over twelve years where he also gained domain knowledge in management educational institutions. He then went on to found a company providing management resource planning systems for educational institutions.',
		# 	'Rishi GK Head Materials',
		# 	'Rishi brings over three decades experience in management products materials logistics. Prior to Integra Rishi held several leadership positions in Government private sectors has also played a vital role in setting up Process Control Systems for the power industry in USA.',
		# 	'P Ravi Head Corporate Affairs HR Policies',
		# 	'With his ability to bring concepts into a clear perspective Ravi brings realization to a vision. As cofounder director for corporate affairs Ravi leads his team in pursuing perfection in all spheres activity with the right amount human touch.',
		# 	'Ravi has national international software development experience in areas such as computer assisted instruction econometric modelling analysis election analysis programming tools project management software plotter interface software financial accounting systems security software. He has also carried out several training programs for corporate customers inhouse engineering staff.',
		# 	'Sandeep Kasliwal Vice President Commercial',
		# 	'Sandeep brings over two decades experience in corporate finance management. Sandeep has held various roles in Integra over the years currently he heads the Commerce Corporate Affairs will be responsible for expanding commerce activities the company.',
		# 	'Rishabh Jain Assistant Vice President',
		# 	'Rishabh is involved in the assessment markets identification realization new growth areas initiation management strategic alliances for the company. His work experience over a decade spans engineering project management marketing sales across domestic international markets.',
		# 	'A.R. Krishna Murthy General Manager Finance',
		# 	'A.R. Krishna Murthy joins Integra with over 37 years experience in finance company affairs. In this span he has extensively held roles in Public Sector Private Sector Government Sector spanning Finance Management Costing General Management areas.',
		# 	'Nitin Shantilal Phuria Chief Software Architect',
		# 	'Nitin joined Integra straight out college 16 years ago has been part the company ever since. He served as a technical architect for Archival Database systems grew proficient in various streams software development at Integra. His current focus is on Financial Inclusion Gateway System AEPS (Aadhaar Enabled Payment System) RuPAY transactions on MicroATM IMPS for Financial Inclusion Unified Payment Interface UIDAI Aadhaar Services for AUA/KUA.',
		# 	'In the past Nitin has successfully architected executed software projects for several African Government organisations which include Tanzania Revenue Authority DataWarehouse Uganda Communications Commission to name some.',
		# 	'Management',
		# 	'Board Advisors',
		# 	'Dr. D K Subramaniam Board Member Advisor',
		# 	"Dr. Subramaniam retired professor dean the Indian Institute Science is a well known visionary in the academic industry circles. He has vast experience in conceptualizing building software solutions is an advisor to many banks financial institutions on IT. He brings his vast experience in the implementation technology to Integra's board.",
		# 	'Dr. V Gopalakrishna Board Member Advisor',
		# 	'Dr. Gopi is passionate about technology true knowledge. As a cofounder the company he ensures that the company does not rest on its laurels continually acquires expertise in new technologies domains. Under his leadership Integra has developed many pioneering products in diverse domains related to networking device drivers imaging embedded systems. He serves as the Managing Director Integra Micro Software Services (P) Ltd (www.integramicroservices.com).',
		# 	'Dr. Gopi was also a founding member Jataayu Software was instrumental in the company becoming a leading provider products solutions in the Wireless internet domain until its acquisition by Comviva Technologies in 2007. His current interests include Web services SOA architecture large enterprise application development using the emerging cloud technologies apart from being actively associated with the Alumni Association Indian Institute Science Bangalore chapter mentoring new entrepreneurial ventures MS/PhD students at the Indian Institute Science Bangalore.',
		# 	"Prior to Integra Dr. Gopi's career spans across National Informatics Centre (NIC) Processor Systems India Canada Centre for Remote Sensing Ottawa in various capacities.",
		# 	'Jyoti Sahai Advisor',
		# 	'Jyoti has more than three decades experience in service organizations. He started as a banker has built many banking software solutions. Jyoti advises Integra on banking solutions operations is the Chairman Managing Director Kavaii Business Analytics India Pvt. Ltd.',
		# 	'Prior to taking up the advisory role at Integra Jyoti has served many prestigious organizations in various capacities from Bank India TCS Birla Horizon International IBM Global Services India vMoksha Technologies as CEO Deutsche Software India.',
		# 	'He regularly blogs on Banking Analytics can be followed on http://jyotisahai.wordpress.com',
		# 	'Raj Kesarimall Advisor',
		# 	'With over two decades experience in the IT Telecom industry having shouldered Sales Marketing Business development PL responsibility across India Europe Americas Raj specializes in setting up new operations market expansion into new geographies building well knit teams to ensure business success.',
		# 	"Prior to the current role Raj was the Vice President Head Europe Americas at Comviva Technologies where he was instrumental in driving the organization's expansion into Latin America setting up Europe operations. Prior to Comviva Technologies Raj was the cofounder Jataayu Software where he played a key role in driving the global expansion the company until its acquisition by Comviva Technologies in 2007.",
		# 	'Dr. V Sudhakar Advisor',
		# 	'Dr Sudhakar Graduate IIT Kharagpur PhD in biophysics from IISc Bangalore worked as a CEO in Tiger Software (a banking software company) for several years. He then later joined Satyam to build products for international markets. He architected 108 emergency ambulance services in Andra Pradesh through PPP involving Police Hospitals Legal machinery to ensure SLA based service to any person in the coverage area using ICT. This has saved thousand lives has now spread to more than 18 states.',
		# 	'Dr Sudhakar then joined Cooptions Technologies Pvt. Ltd. a rural facilitator in the villages for accessing formal financial aids available for a common man. He was instrumental in securing 8M USD investments from International markets.',
		# 	'Dr. Sudhakar served as CEO Namma Bengaluru Foundation in the past. Currently he is involved in helping social enterprises like i25 RMCS; he is spending significant time pursing education for emerging entrepreneurs in developing countries (an international institution in Bangalore).',
		# 	'Rajeev Bhagwat Advisor',
		# 	'Rajeev advises Integra on business strategies administration management finance. He is the CEO XL Management services has been associated with Integra from its early years.',
		# 	'About Us',
		# 	'Why Integra?',
		# 	'Leadership',
		# 	'Partners Customers',
		# 	'Affiliated Companies',
		# 	'CSR',
		# 	'Contact Us',
		# 	'/',
		# 	'Group Companies',
		# 	'Integra Micro Software Services (P) Ltd.',
		# 	'Mobile Apps Enterprise Solutions Mobile Finance BPM K2 Integrated Testing Consulting Outsourcing.',
		# 	'Jakkur Technoparks Private Limited.',
		# 	'Provider insurance services',
		# 	'Partner Company',
		# 	'i25 Rural Mobile Commerce Services',
		# 	'i25 Rural Mobile Commerce Services (i25 RMCS) provides last mile operations in urban semiurban rural areas.',
		# 	'Integra Micro Systems (P) Ltd.',
		# 	'Integra is a leading provider innovative hitechnology products solutions in the Government BFSI telecom space with a focus on India Africa.',
		# 	'Contact Us',
		# 	'Integra Micro Systems (P) Ltd.',
		# 	'#4 Bellary Road 12th KM post Jakkur Bangalore 560064 India.',
		# 	'Tel: +9180 28565801 (23)',
		# 	'Email: info@integramicro.com',
		# 	'For Payments click here.',
		# 	'Follow Us',
		# 	'Twitter',
		# 	'Facebook',
		# 	'Linkedin',
		# 	'Enquiry Form',
		# 	'Select State Andaman Nicobar Islands Andhra Pradesh Arunachal Pradesh Assam Bihar Chandigarh Chhattisgarh Delhi Goa Gujarat Haryana Himachal Pradesh Jharkhand Karnataka Kerala Madhya Pradesh Maharashtra Manipur Meghalaya Mizoram Nagaland Orissa Pondicherry Punjab Rajasthan Sikkim Tamil Nadu Telangana Tripura Uttarakhand Uttar Pradesh West Bengal Others',
		# 	'© 2016 Integra Micro Systems (P) Ltd.',
		# 	'All product names logos brands are the property their respective owners.',
		# 	'All company product service names used in this website are for identification purposes only.'
		# 	]
		self.text_len = len(self.texts)